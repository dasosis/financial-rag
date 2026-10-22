"""
The real langchain models through the real gateway, against its fake OpenAI.

Opt-in: it builds the gateway and the fake upstream with Go (gateway/), and
runs both on loopback ports. Nothing reaches the internet or costs money.

    RUN_GATEWAY_TESTS=1 pytest tests/test_gateway_live.py

GO names the go binary if it isn't on PATH.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
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
def gateway(tmp_path_factory):
    go = os.getenv("GO") or shutil.which("go")
    if not go:
        pytest.skip("Go isn't installed (set GO to its path)")
    out = tmp_path_factory.mktemp("bin")
    exe = ".exe" if sys.platform == "win32" else ""
    binaries = {}
    for name in ("gateway", "fakeupstream"):
        binaries[name] = out / (name + exe)
        subprocess.run([go, "build", "-o", str(binaries[name]), f"./cmd/{name}"],
                       cwd=GATEWAY_DIR, check=True)

    fake_port, port, metrics_port = _free_port(), _free_port(), _free_port()
    fake = subprocess.Popen([str(binaries["fakeupstream"]), "-addr", f"127.0.0.1:{fake_port}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env = {**os.environ, "LLM_GATEWAY_KEY": KEY, "OPENAI_API_KEY": "sk-fake-upstream",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{fake_port}/v1",
           "GATEWAY_ADDR": f"127.0.0.1:{port}", "GATEWAY_METRICS_ADDR": f"127.0.0.1:{metrics_port}"}
    server = subprocess.Popen([str(binaries["gateway"])], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait(f"http://127.0.0.1:{port}/healthz")
        yield {"url": f"http://127.0.0.1:{port}/v1",
               "metrics": f"http://127.0.0.1:{metrics_port}/metrics"}
    finally:
        server.terminate()
        fake.terminate()
        server.wait(timeout=10)
        fake.wait(timeout=10)


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


def test_a_wrong_key_is_refused(monkeypatch):
    import openai

    monkeypatch.setenv("LLM_GATEWAY_KEY", "not-the-key")
    with pytest.raises(openai.AuthenticationError):
        config.chat_model(max_retries=0).invoke("hi")
