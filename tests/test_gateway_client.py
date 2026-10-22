"""Tests for calling models through the LLM gateway (config.py), offline.

Without LLM_GATEWAY_URL nothing changes: ChatOpenAI and OpenAIEmbeddings are
built as before, straight to OpenAI. With it, the gateway's HTTP responses are
played by a mock transport, so no Go, Redis or network is needed.
"""

import json

import httpx2
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from financial_rag import api, config, jobs, router, tracing, usage

ALICE = "0f8b1c2e-alice"
GATEWAY = "http://gateway.test:8080/v1"


@pytest.fixture(autouse=True)
def fresh_config(monkeypatch):
    """Each test starts with the gateway off and the process consumer unnamed."""
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    monkeypatch.setattr(config, "_process_consumer", "system:cli")
    monkeypatch.setattr(config, "_cache_responses", True)
    config._gateway_client.cache_clear()
    yield
    config._gateway_client.cache_clear()


class Gateway:
    """Plays the gateway: records each request and answers from `replies`."""

    def __init__(self):
        self.requests: list[httpx2.Request] = []
        self.replies: list[httpx2.Response] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.replies:
            return self.replies.pop(0)
        if request.url.path.endswith("/embeddings"):
            texts = json.loads(request.content)["input"]
            return httpx2.Response(200, json={
                "object": "list", "model": config.EMBEDDING_MODEL,
                "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]}
                         for i, _ in enumerate(texts)],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            })
        return httpx2.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
            "model": config.CHAT_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Hello."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        })

    def body(self, n: int = -1) -> dict:
        return json.loads(self.requests[n].content)


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_URL", GATEWAY + "/")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "internal-key-for-tests")
    fake = Gateway()

    class Mocked(config._GatewayClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, http_client=httpx2.Client(
                transport=httpx2.MockTransport(fake.handle)))

    monkeypatch.setattr(config, "_GatewayClient", Mocked)
    return fake


def refusal(status: int, code: str, retry_after: str | None = "120") -> httpx2.Response:
    headers = {"x-should-retry": "false"}
    if retry_after:
        headers["retry-after"] = retry_after
    return httpx2.Response(status, headers=headers, json={
        "error": {"message": "refused", "type": "gateway_error", "code": code}})


def exception(cls, retry_after="30"):
    """A gateway refusal as the client would raise it."""
    request = httpx2.Request("POST", GATEWAY + "/chat/completions")
    response = httpx2.Response(429 if issubclass(cls, config.GatewayLimit) else 503,
                               headers={"retry-after": retry_after}, request=request)
    return cls(response)


# ---------------------------------------------------------------------------
# Gateway off: exactly as before
# ---------------------------------------------------------------------------

def test_without_the_gateway_models_are_built_as_before():
    llm = config.chat_model(temperature=0, timeout=20, max_retries=1)
    assert isinstance(llm, ChatOpenAI)
    plain = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, timeout=20, max_retries=1)
    assert llm.model_dump() == plain.model_dump()
    assert type(llm.root_client) is openai.OpenAI
    assert llm.openai_api_base is None and llm.stream_usage is True

    emb = config.embeddings()
    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.model_dump() == OpenAIEmbeddings(model=config.EMBEDDING_MODEL).model_dump()


def test_without_the_gateway_no_consumer_header_is_sent():
    llm = config.chat_model()
    assert "X-Consumer-Id" not in llm.root_client.default_headers


# ---------------------------------------------------------------------------
# Gateway on
# ---------------------------------------------------------------------------

def test_calls_go_to_the_gateway_with_its_key_and_the_consumer(gateway):
    llm = config.chat_model(temperature=0)
    assert llm.invoke("hi").content == "Hello."
    request = gateway.requests[0]
    assert str(request.url) == GATEWAY + "/chat/completions"
    assert request.headers["authorization"] == "Bearer internal-key-for-tests"
    assert request.headers["x-consumer-id"] == "system:cli"
    assert "cache-control" not in request.headers
    assert gateway.body()["model"] == config.CHAT_MODEL


def test_the_consumer_is_read_per_call_in_the_calling_context(gateway):
    llm = config.chat_model(temperature=0)
    with config.as_consumer("user:alice"):
        llm.invoke("hi")
    llm.invoke("hi")
    assert [r.headers["x-consumer-id"] for r in gateway.requests] == ["user:alice", "system:cli"]


def test_scripts_name_themselves_and_benchmarks_bypass_the_cache(gateway):
    config.run_as("system:eval", cache=False)
    config.chat_model(temperature=0).invoke("hi")
    assert gateway.requests[0].headers["x-consumer-id"] == "system:eval"
    assert gateway.requests[0].headers["cache-control"] == "no-cache"


def test_streams_ask_for_usage_as_against_openai(gateway):
    gateway.replies.append(httpx2.Response(
        200, headers={"content-type": "text/event-stream"},
        content=(b'data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m",'
                 b'"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
                 b"data: [DONE]\n\n")))
    parts = list(config.chat_model(temperature=0).stream("hi"))
    assert "".join(p.content for p in parts) == "Hi"
    assert gateway.body()["stream_options"] == {"include_usage": True}


def test_embeddings_send_text_not_token_ids(gateway):
    vectors = config.embeddings().embed_documents(["Total revenue 245,122", "Net income"])
    assert len(vectors) == 2
    body = gateway.body()
    assert body["input"] == ["Total revenue 245,122", "Net income"]
    assert body["model"] == config.EMBEDDING_MODEL
    assert gateway.requests[0].headers["x-consumer-id"] == "system:cli"


def test_a_missing_gateway_key_is_an_error(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_URL", GATEWAY)
    with pytest.raises(RuntimeError, match="LLM_GATEWAY_KEY"):
        config.chat_model()


@pytest.mark.parametrize(("status", "code", "cls"), [
    (429, "budget_exhausted", config.BudgetExhausted),
    (429, "rate_limited", config.RateLimited),
    (503, "providers_unavailable", config.ProvidersUnavailable),
])
def test_refusals_raise_their_own_exceptions_without_retrying(gateway, status, code, cls):
    gateway.replies.append(refusal(status, code))
    with pytest.raises(cls) as caught:
        config.chat_model(temperature=0, max_retries=2).invoke("hi")
    assert len(gateway.requests) == 1      # x-should-retry: false
    assert caught.value.status_code == status
    assert "refused" not in str(caught.value)     # a message for the user instead
    if status == 429:
        assert caught.value.retry_after == 120


def test_a_rate_limit_message_says_when_to_retry(gateway):
    gateway.replies.append(refusal(429, "rate_limited", retry_after="7"))
    with pytest.raises(config.RateLimited, match="7 seconds"):
        config.chat_model(max_retries=0).invoke("hi")


def test_other_errors_are_the_usual_openai_errors(gateway):
    gateway.replies.append(httpx2.Response(400, json={
        "error": {"message": "bad", "type": "invalid_request_error", "code": "invalid_body"}}))
    with pytest.raises(openai.BadRequestError) as caught:
        config.chat_model(max_retries=0).invoke("hi")
    assert not isinstance(caught.value, config.GatewayRefusal)


def test_refusals_pass_through_structured_output_unchanged(gateway):
    """with_structured_output, as the planner and reranker use it, re-raises
    OpenAI errors as langchain's own; the gateway's must arrive as they are."""
    gateway.replies.append(refusal(429, "budget_exhausted"))
    planner = config.chat_model(temperature=0).with_structured_output(router.QuestionPlan)
    with pytest.raises(config.BudgetExhausted):
        planner.invoke([("human", "Operating margin?")])


def test_embedding_refusals_raise_too(gateway):
    gateway.replies.append(refusal(429, "budget_exhausted"))
    with pytest.raises(config.BudgetExhausted):
        config.embeddings().embed_query("revenue")


# ---------------------------------------------------------------------------
# The API: the right consumer, and refusals as the usual 429
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    tracing.setup()
    return TestClient(api.app)


def _consumer_recording_answer(seen):
    def answer(question, owner, source_file=None, history=None, **kw):
        seen.append(config.consumer())
        return router.Answer("An answer.", [], "text", question=question)
    return answer


def test_a_question_is_asked_as_its_user(client, monkeypatch):
    seen = []
    monkeypatch.setattr(router, "answer", _consumer_recording_answer(seen))
    assert client.post("/query", json={"question": "Revenue?"},
                       headers={"X-User-Id": ALICE}).status_code == 200
    client.post("/query/stream", json={"question": "Revenue?"}, headers={"X-User-Id": ALICE})
    assert seen == [f"user:{ALICE}", f"user:{ALICE}"]
    assert config.consumer() == "system:cli"     # and nothing leaks out of the request


def test_a_job_runs_as_its_owner():
    seen = []
    job = jobs.submit(ALICE, "test", "x", work=lambda step: seen.append(config.consumer()) or "{}")
    jobs._executor.submit(lambda: None).result(timeout=10)
    assert jobs.get(job.id).state == jobs.DONE
    assert seen == [f"user:{ALICE}"]


@pytest.mark.parametrize("cls", [config.BudgetExhausted, config.RateLimited])
def test_a_gateway_limit_is_the_usual_429_and_refunded(client, monkeypatch, cls):
    def limited(*args, **kwargs):
        raise exception(cls, retry_after="45")

    monkeypatch.setattr(router, "answer", limited)
    response = client.post("/query", json={"question": "Revenue?"}, headers={"X-User-Id": ALICE})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "45"
    assert response.json() == {"detail": str(exception(cls, retry_after="45"))}
    assert usage.usage(ALICE).questions == 0


def test_a_gateway_limit_while_streaming_is_an_error_event_and_refunded(client, monkeypatch):
    def limited(*args, **kwargs):
        raise exception(config.BudgetExhausted)

    monkeypatch.setattr(router, "answer", limited)
    response = client.post("/query/stream", json={"question": "Revenue?"},
                           headers={"X-User-Id": ALICE})
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[-1]["type"] == "error" and events[-1]["status"] == 429
    assert "allowance" in events[-1]["detail"]
    assert usage.usage(ALICE).questions == 0


def test_no_provider_is_a_503_and_refunded(client, monkeypatch):
    def down(*args, **kwargs):
        raise exception(config.ProvidersUnavailable)

    monkeypatch.setattr(router, "answer", down)
    response = client.post("/query", json={"question": "Revenue?"}, headers={"X-User-Id": ALICE})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
    assert usage.usage(ALICE).questions == 0


# ---------------------------------------------------------------------------
# Fallbacks degrade on provider errors, never on a spent budget
# ---------------------------------------------------------------------------

def test_the_follow_up_rewrite_doesnt_swallow_a_limit(monkeypatch):
    from financial_rag import followup

    def failing(error):
        class Broken:
            def __init__(self, **kwargs):
                pass

            def with_structured_output(self, schema):
                return self

            def invoke(self, messages):
                raise error
        return Broken

    import langchain_openai
    history = [("user", "Revenue in 2024?")]
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", failing(exception(config.BudgetExhausted)))
    with pytest.raises(config.BudgetExhausted):
        followup.standalone_question("And 2023?", history)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI",
                        failing(exception(config.ProvidersUnavailable)))
    assert followup.standalone_question("And 2023?", history) == "And 2023?"
