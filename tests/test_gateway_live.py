"""
The real langchain models through the real gateway, against its fake OpenAI.

Opt-in: it builds the gateway and the fake upstream with Go (gateway/), and
runs both on loopback ports. Nothing reaches the internet or costs money.

    RUN_GATEWAY_TESTS=1 pytest tests/test_gateway_live.py

GO names the go binary if it isn't on PATH. The response cache tests need a
Redis: GATEWAY_TEST_REDIS_URL=redis://localhost:6379/1.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

from financial_rag import config, followup, router

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_GATEWAY_TESTS") != "1", reason="set RUN_GATEWAY_TESTS=1 (needs Go)")

GATEWAY_DIR = Path(__file__).resolve().parent.parent / "gateway"
KEY = "integration-test-key-0123456789"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1).close()
            return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"{url} never answered")


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    go = os.getenv("GO") or shutil.which("go")
    if not go:
        pytest.skip("Go isn't installed (set GO to its path)")
    out = tmp_path_factory.mktemp("bin")
    exe = ".exe" if sys.platform == "win32" else ""
    built = {}
    for name in ("gateway", "fakeupstream"):
        built[name] = out / (name + exe)
        subprocess.run([go, "build", "-o", str(built[name]), f"./cmd/{name}"],
                       cwd=GATEWAY_DIR, check=True)
    fake_port = _free_port()
    fake = subprocess.Popen([str(built["fakeupstream"]), "-addr", f"127.0.0.1:{fake_port}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    built["upstream"] = f"http://127.0.0.1:{fake_port}/v1"
    yield built
    fake.terminate()
    fake.wait(timeout=10)


@contextmanager
def _running_gateway(binaries, **settings):
    """A gateway on loopback ports in front of the fake upstream. No Redis: its
    limits are its own (GATEWAY_LIMITS_REDIS_URL unset)."""
    port, metrics_port = _free_port(), _free_port()
    env = {**os.environ, "LLM_GATEWAY_KEY": KEY, "OPENAI_API_KEY": "sk-fake-upstream",
           "OPENAI_BASE_URL": binaries["upstream"],
           "GATEWAY_ADDR": f"127.0.0.1:{port}", "GATEWAY_METRICS_ADDR": f"127.0.0.1:{metrics_port}",
           **settings}
    server = subprocess.Popen([str(binaries["gateway"])], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait(f"http://127.0.0.1:{port}/healthz")
        yield {"url": f"http://127.0.0.1:{port}/v1",
               "metrics": f"http://127.0.0.1:{metrics_port}/metrics"}
    finally:
        server.terminate()
        server.wait(timeout=10)


@pytest.fixture(scope="module")
def gateway(binaries):
    with _running_gateway(binaries) as running:
        yield running


@pytest.fixture(autouse=True)
def through_the_gateway(gateway, monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_URL", gateway["url"])
    monkeypatch.setenv("LLM_GATEWAY_KEY", KEY)
    monkeypatch.setattr(config, "_process_consumer", "system:cli")
    config._gateway_client.cache_clear()
    yield
    config._gateway_client.cache_clear()


def _metrics(gateway) -> str:
    with urllib.request.urlopen(gateway["metrics"], timeout=5) as response:
        return response.read().decode()


def test_invoke(gateway):
    answer = config.chat_model(temperature=0).invoke("What was revenue?")
    assert answer.content.startswith("Fake answer")
    assert answer.usage_metadata["input_tokens"] > 0
    assert 'gateway_requests_total{code="200",endpoint="chat"}' in _metrics(gateway)


def test_stream(gateway):
    chunks = list(config.chat_model(temperature=0).stream("What was revenue?"))
    assert len(chunks) > 3
    text = "".join(c.content for c in chunks)
    assert text.startswith("Fake answer") and text.endswith("characters.")
    # ChatOpenAI asked for usage (as it does against OpenAI), so it got it
    total = chunks[0]
    for chunk in chunks[1:]:
        total += chunk
    assert total.usage_metadata["output_tokens"] > 0


def test_with_structured_output():
    plan = config.chat_model(temperature=0).with_structured_output(router.QuestionPlan).invoke(
        [("human", "What was the operating margin in 2024?")])
    assert isinstance(plan, router.QuestionPlan)
    # And the follow-up rewrite, end to end through its own structured output
    rewritten = followup.standalone_question("And 2023?", [("user", "Revenue in 2024?")])
    assert rewritten == "fake"


def test_embeddings():
    vectors = config.embeddings().embed_documents(["Total revenue 245,122", "Net income"])
    assert [len(v) for v in vectors] == [1536, 1536]
    assert vectors[0] != vectors[1]
    assert config.embeddings().embed_query("Net income") == vectors[1]


def _use(gateway, monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_URL", gateway["url"])
    config._gateway_client.cache_clear()


def test_a_spent_budget_reaches_the_app_as_budget_exhausted(binaries, monkeypatch):
    with _running_gateway(binaries, GATEWAY_DAILY_BUDGET_USER="1500",
                          GATEWAY_COMPLETION_ESTIMATE="1", GATEWAY_RATE_PER_CONSUMER="0",
                          GATEWAY_RATE_GLOBAL="0") as limited:
        _use(limited, monkeypatch)
        llm = config.chat_model(temperature=0)
        with config.as_consumer("user:alice"):
            llm.invoke("What was revenue?")       # charged its real usage, ~60 units
            with pytest.raises(config.BudgetExhausted) as caught:
                for _ in range(100):
                    llm.invoke("What was revenue?")
        assert 0 < caught.value.retry_after <= 86_400
        # Another consumer still has its own budget
        with config.as_consumer("user:bob"):
            assert llm.invoke("hi").content.startswith("Fake answer")


def test_a_rate_limit_reaches_the_app_as_rate_limited(binaries, monkeypatch):
    with _running_gateway(binaries, GATEWAY_RATE_PER_CONSUMER="1",
                          GATEWAY_BURST_PER_CONSUMER="2") as limited:
        _use(limited, monkeypatch)
        llm = config.chat_model(temperature=0, max_retries=2)
        with config.as_consumer("user:alice"), pytest.raises(config.RateLimited) as caught:
            for _ in range(3):
                llm.invoke("hi")
        assert 50 <= caught.value.retry_after <= 60
        assert "gateway_limit_rejections_total{reason=\"consumer_rate\"} 1" in _metrics(limited)


@pytest.fixture
def cached_gateway(binaries, monkeypatch):
    url = os.getenv("GATEWAY_TEST_REDIS_URL")
    if not url:
        pytest.skip("set GATEWAY_TEST_REDIS_URL for the response cache tests")
    with _running_gateway(binaries, GATEWAY_CACHE_REDIS_URL=url) as running:
        _use(running, monkeypatch)
        yield running


def _cache_counts(gateway) -> dict[str, float]:
    counts = {}
    for line in _metrics(gateway).splitlines():
        if line.startswith("gateway_cache_requests_total{"):
            result = line.split('result="')[1].split('"')[0]
            counts[result] = float(line.rsplit(" ", 1)[1])
    return counts


def test_a_repeat_is_answered_from_the_cache(cached_gateway):
    question = f"What was revenue? {time.time_ns()}"   # unique per run: the Redis persists
    llm = config.chat_model(temperature=0)
    first, second = llm.invoke(question), llm.invoke(question)
    assert second.content == first.content
    assert _cache_counts(cached_gateway) == {"hit": 1, "miss": 1, "bypass": 0}
    # A streamed repeat is replayed as SSE, with the usage ChatOpenAI asks for
    chunks = list(llm.stream(question))
    assert "".join(c.content for c in chunks) == first.content
    total = chunks[0]
    for chunk in chunks[1:]:
        total += chunk
    assert total.usage_metadata["output_tokens"] > 0
    assert _cache_counts(cached_gateway)["hit"] == 2


def test_structured_output_from_the_cache(cached_gateway):
    planner = config.chat_model(temperature=0).with_structured_output(router.QuestionPlan)
    question = [("human", f"Operating margin in 2024? {time.time_ns()}")]
    assert planner.invoke(question) == planner.invoke(question)
    assert _cache_counts(cached_gateway)["hit"] == 1


def test_benchmarks_bypass_the_cache(cached_gateway, monkeypatch):
    monkeypatch.setattr(config, "_cache_responses", False)   # as run_as(..., cache=False)
    question = f"What was revenue? {time.time_ns()}"
    llm = config.chat_model(temperature=0)
    llm.invoke(question)
    llm.invoke(question)
    assert _cache_counts(cached_gateway) == {"hit": 0, "miss": 0, "bypass": 2}


def test_with_openai_down_the_backup_answers(binaries, monkeypatch):
    """OpenAI unreachable, the backup (the same fake, as OpenRouter) answers:
    invoke, stream, and structured output, which the gateway checks against
    the schema before it goes out. Embeddings have no backup: 503."""
    with _running_gateway(binaries, OPENAI_BASE_URL=f"http://127.0.0.1:{_free_port()}/v1",
                          OPENROUTER_API_KEY="sk-or-fake",
                          OPENROUTER_BASE_URL=binaries["upstream"]) as gateway:
        _use(gateway, monkeypatch)
        llm = config.chat_model(temperature=0)
        assert llm.invoke("What was revenue?").content.startswith("Fake answer")
        assert "".join(c.content for c in llm.stream("What was revenue?")).startswith("Fake answer")
        plan = llm.with_structured_output(router.QuestionPlan).invoke([("human", "Operating margin?")])
        assert isinstance(plan, router.QuestionPlan)
        with pytest.raises(config.ProvidersUnavailable):
            config.embeddings().embed_query("Net income")
        assert 'gateway_served_total{endpoint="chat",provider="openrouter"} 3' in _metrics(gateway)


def test_a_wrong_key_is_refused(monkeypatch):
    import openai

    monkeypatch.setenv("LLM_GATEWAY_KEY", "not-the-key")
    with pytest.raises(openai.AuthenticationError):
        config.chat_model(max_retries=0).invoke("hi")
