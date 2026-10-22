"""The LLM gateway's redaction test data stays in step with the app.

gateway/internal/redact checks its detectors against every benchmark question
and both sample 10-Ks, from fixtures; these tests fail when the fixtures no
longer match eval_answers.py (always) or the filings in data/ (when present).
Regenerate with: python gateway/internal/redact/testdata/make_fixtures.py
"""

import gzip
import importlib.util
from pathlib import Path

import pytest

TESTDATA = Path(__file__).resolve().parent.parent / "gateway" / "internal" / "redact" / "testdata"


def _maker():
    spec = importlib.util.spec_from_file_location("make_fixtures", TESTDATA / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_question_fixture_matches_the_benchmark():
    maker = _maker()
    assert maker.QUESTIONS.read_text(encoding="utf-8") == maker.eval_questions(), (
        "eval_answers.py changed: regenerate the gateway's redaction fixtures")


@pytest.mark.parametrize("source", ["microsoft-sec-10k.htm", "apple-sec-10k.htm"])
def test_the_filing_fixtures_match_the_samples(source):
    maker = _maker()
    path = maker.ROOT / "data" / source
    if not path.exists():
        pytest.skip(f"{source} isn't in data/ (python -m financial_rag.ingest --fetch-samples)")
    fixture = gzip.decompress((TESTDATA / maker.FILINGS[source]).read_bytes()).decode("utf-8")
    assert fixture == maker.filing_text(path)
