"""
config.py — The models the pipeline uses, named in one place, and how to reach them.

Every language-model call (planning, follow-up rewriting, figure extraction,
LLM reranking, answering, the answer retry) uses CHAT_MODEL. Set the
CHAT_MODEL environment variable to try another model, and measure the change
with eval_answers.py rather than guessing:

    CHAT_MODEL=gpt-5-mini python -m financial_rag.eval_answers

Every model is built by chat_model() or embeddings(). They call OpenAI directly,
unless LLM_GATEWAY_URL names the LLM gateway (gateway/, see the README). Then
every call goes there instead, with LLM_GATEWAY_KEY as its key (the OpenAI key
lives only in the gateway) and an X-Consumer-Id header saying whose call it is:
"user:<id>" for a signed-in user's question or upload (see as_consumer), or a
system name for a script ("system:ingest", "system:eval"; see run_as). The
gateway enforces budgets and rate limits per consumer, and its refusals arrive
as the exceptions below.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import openai

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Not overridable: the vector store holds vectors from this model, so changing
# it means re-ingesting everything (python start.py --reingest).
EMBEDDING_MODEL = "text-embedding-3-small"

# The Dockerfile bakes this model into the image; keep its RUN line in sync.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def gateway_url() -> str | None:
    """The LLM gateway's base URL (".../v1"), or None to call OpenAI directly."""
    return os.getenv("LLM_GATEWAY_URL", "").strip().rstrip("/") or None


def chat_model(**kwargs):
    """A ChatOpenAI for CHAT_MODEL (unless `model` says otherwise), with these options."""
    from langchain_openai import ChatOpenAI

    kwargs.setdefault("model", CHAT_MODEL)
    url = gateway_url()
    if url is None:
        return ChatOpenAI(**kwargs)
    client = _gateway_client(url, _gateway_key(), kwargs.get("timeout"), kwargs.get("max_retries"))
    # Against OpenAI itself, ChatOpenAI asks for token usage in streams by
    # default; with a client of our own it wouldn't, so ask as it would.
    kwargs.setdefault("stream_usage", True)
    return ChatOpenAI(**kwargs, base_url=url, api_key=_gateway_key(),
                      root_client=client, client=client.chat.completions)


def embeddings():
    """The OpenAIEmbeddings the vector store was built with (EMBEDDING_MODEL)."""
    from langchain_openai import OpenAIEmbeddings

    url = gateway_url()
    if url is None:
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    client = _gateway_client(url, _gateway_key(), None, None)
    # Send the text itself rather than token ids, so the gateway can read what it
    # forwards. Chunks are far below the model's input limit, which is all the
    # token-id path guards against; the vectors are the same.
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, base_url=url, api_key=_gateway_key(),
                            client=client.embeddings, check_embedding_ctx_length=False)


def _gateway_key() -> str:
    key = os.getenv("LLM_GATEWAY_KEY", "").strip()
    if not key:
        raise RuntimeError("LLM_GATEWAY_URL is set but LLM_GATEWAY_KEY isn't.")
    return key


# ---------------------------------------------------------------------------
# Whose call it is: the X-Consumer-Id header
# ---------------------------------------------------------------------------

# A request's user, set by the API around the work it does for them. Threads
# don't inherit context variables, so threads that do a user's work set it
# themselves (api.py's streaming worker, jobs.py's jobs).
_consumer: ContextVar[str | None] = ContextVar("llm_consumer", default=None)
# Everything else: the script that is running (run_as), or the command line
_process_consumer = "system:cli"
_cache_responses = True


def consumer() -> str:
    """The consumer the current model call is made for."""
    return _consumer.get() or _process_consumer


@contextmanager
def as_consumer(name: str) -> Iterator[None]:
    """Within this block, in this thread, model calls are made for `name`."""
    token = _consumer.set(name)
    try:
        yield
    finally:
        _consumer.reset(token)


def user_consumer(owner: str) -> str:
    return f"user:{owner}"


def run_as(name: str, cache: bool = True) -> None:
    """
    Name the script running ("system:ingest") as the consumer of its model calls.
    cache=False asks the gateway never to answer from its response cache, as
    benchmarks must: a cached answer would score the cache, not the model.
    """
    global _process_consumer, _cache_responses
    _process_consumer = name
    _cache_responses = cache


# ---------------------------------------------------------------------------
# The gateway's refusals, as exceptions
# ---------------------------------------------------------------------------

class GatewayRefusal(Exception):
    """The gateway refused a call. Deliberately not an openai.APIError: ChatOpenAI
    re-raises those as its own types, and these must reach the API as they are."""

    def __init__(self, response) -> None:
        self.response = response
        self.status_code = response.status_code
        super().__init__(self.describe())

    def describe(self) -> str:
        raise NotImplementedError


class GatewayLimit(GatewayRefusal):
    """A budget or rate limit was reached (429).

    Never swallow one in a fallback: the API turns it into its usual 429 for the
    user, and refunds the question.
    """

    @property
    def retry_after(self) -> int:
        """Seconds until it's worth asking again."""
        try:
            return max(1, int(float(self.response.headers.get("retry-after", "60"))))
        except (TypeError, ValueError):
            return 60


class BudgetExhausted(GatewayLimit):
    def describe(self) -> str:
        return ("You've used today's allowance for the language model. You can ask "
                "again after midnight UTC.")


class RateLimited(GatewayLimit):
    def describe(self) -> str:
        return (f"The app is handling too many questions right now. Try again in "
                f"{self.retry_after} seconds.")


class ProvidersUnavailable(GatewayRefusal):
    """No model provider could take the call (503). Fallbacks may degrade on it,
    as they do on any other failed call."""

    def describe(self) -> str:
        return "The language model is unavailable right now. Please try again in a few minutes."


_GATEWAY_ERRORS: dict[str, type[GatewayRefusal]] = {
    "budget_exhausted": BudgetExhausted,
    "rate_limited": RateLimited,
    "providers_unavailable": ProvidersUnavailable,
}


class _GatewayClient(openai.OpenAI):
    """The OpenAI client, pointed at the gateway: it names the consumer on every
    request, and raises the gateway's refusals as the exceptions above."""

    @property
    def default_headers(self) -> dict:
        # Read per request, in the thread making it, so one client serves everyone
        headers = {**super().default_headers, "X-Consumer-Id": consumer()}
        if not _cache_responses:
            headers["Cache-Control"] = "no-cache"
        return headers

    def _make_status_error(self, err_msg, *, body, response):
        data = body.get("error", body) if isinstance(body, dict) else body
        code = data.get("code") if isinstance(data, dict) else None
        error = _GATEWAY_ERRORS.get(code) if isinstance(code, str) else None
        if error is not None:
            return error(response)   # raised by the client in place of its own
        return super()._make_status_error(err_msg, body=body, response=response)


@lru_cache(maxsize=16)
def _gateway_client(url: str, key: str, timeout, max_retries) -> _GatewayClient:
    """One client (and connection pool) per gateway and set of call options. The
    timeout and retries are passed as ChatOpenAI passes them to its own client."""
    options = {} if max_retries is None else {"max_retries": max_retries}
    return _GatewayClient(api_key=key, base_url=url, timeout=timeout, **options)
