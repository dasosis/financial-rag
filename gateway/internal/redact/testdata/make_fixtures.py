"""
Write the redactor's test data: the text of both sample 10-Ks as the app reads
them (html_filing.load_pages, what gets chunked and sent to the model), and
every eval_answers question, follow-up history included.

The redact package's tests run the redactor over all of it and fail on any
change to a figure, year, EPS value, share count or citation. Regenerate after
changing the samples or the benchmark, from the repository root, with the
samples in data/ (python -m financial_rag.ingest --fetch-samples):

    python gateway/internal/redact/testdata/make_fixtures.py

tests/test_redaction_fixtures.py fails when the questions here fall out of
step with eval_answers.py.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FILINGS = {"microsoft-sec-10k.htm": "microsoft-10k.txt.gz", "apple-sec-10k.htm": "apple-10k.txt.gz"}
QUESTIONS = HERE / "eval-questions.txt"


def filing_text(path: Path) -> str:
    from financial_rag import html_filing

    html = path.read_text(encoding="utf-8", errors="replace")
    return "\n\n".join(page.text for page in html_filing.load_pages(html))


def eval_questions() -> str:
    """One text per line: each question, then each turn of its history."""
    from financial_rag import eval_answers

    lines = []
    for case in eval_answers.CASES:
        lines.append(case.question)
        lines.extend(text for _, text in case.history)
    return "\n".join(" ".join(line.split()) for line in lines) + "\n"


def main() -> None:
    sys.path.insert(0, str(ROOT))
    QUESTIONS.write_text(eval_questions(), encoding="utf-8", newline="\n")
    for source, fixture in FILINGS.items():
        text = filing_text(ROOT / "data" / source)
        # mtime=0: the same text always gives the same bytes
        (HERE / fixture).write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        print(f"{fixture}: {len(text):,} characters")
    print(f"{QUESTIONS.name}: {len(eval_questions().splitlines())} texts")


if __name__ == "__main__":
    main()
