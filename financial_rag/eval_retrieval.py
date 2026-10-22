"""
eval_retrieval.py — Retrieval accuracy of the pipeline's configurations.

A question scores a hit when a retrieved chunk contains its evidence: the figure
or phrase from the ground truth, verbatim as the filing prints it. Reports
hit@1, hit@TOP_K and mean reciprocal rank over three sets of questions:

  - the evaluate.py questions (plain-English, "fiscal year 2024")
  - exact-term questions: quoted values, line-item labels, "FY24"
  - more questions: older fiscal years, other line items, segments, notes and
    management's discussion, written before any configuration was scored

and three configurations:

  - semantic search + cross-encoder (the original pipeline)
  - hybrid search + cross-encoder
  - hybrid search + LLM reranking (the default)

Every answer was checked against Microsoft's FY2024 10-K text. The LLM
configuration makes one gpt-4o-mini call per question (about $0.002 each, so
roughly $0.15 a run); the others only embed the questions. Needs the ingested
sample (python -m financial_rag.ingest). Pass --no-llm to skip the LLM configuration.

Usage:
    python -m financial_rag.eval_retrieval [--no-llm]
"""

from __future__ import annotations

import re
import sys

from financial_rag import config, rag
from financial_rag.evaluate import EVAL_QUESTIONS
from financial_rag.ingest import PUBLIC_OWNER

# Evidence for ground truths without a figure in them; None = not scored
EVIDENCE_OVERRIDES = {
    "Who is the company's independent registered public accounting firm?": "Deloitte & Touche",
    "What are the company's reportable segments?": None,
    "What are the main risk factors disclosed in the filing?": None,
}

# Figures from the evaluate.py ground truths, asked the way an analyst types them
EXACT_QUESTIONS = [
    ("Which balance sheet line was $57,582 million at June 30, 2024?", "57,582"),
    ("Where does the filing report 245,122?", "245,122"),
    ("FY24 additions to property and equipment", "44,477"),
    ("Diluted EPS FY24", "11.80"),
    ("What was \"Net cash from operations\" in fiscal 2024?", "118,548"),
    ("Common stock cash dividends paid FY2024", "21,771"),
    ("Intelligent Cloud revenue FY24", "105,362"),
    ("Stock-based compensation expense FY24", "10,734"),
    ("Goodwill at June 30, 2024", "119,220"),
    ("Short-term unearned revenue FY24", "57,582"),
]

# Evidence is a string, or a tuple of strings any of which counts (the same fact
# printed two ways in different sections)
MORE_QUESTIONS: list[tuple[str, str | tuple[str, ...]]] = [
    # Older fiscal years and other line items, phrased several ways
    ("What was Microsoft's revenue in fiscal 2022?", "198,270"),
    ("How much did Microsoft spend on R&D in FY2023?", "27,195"),
    ("General and administrative expense for the year ended June 30, 2022", "5,900"),
    ("What was other income (expense), net in fiscal year 2024?", "1,646"),
    ("Income before income taxes FY23", "89,311"),
    ("Basic earnings per share for fiscal 2022", "9.70"),
    ("How many diluted weighted average shares were outstanding in fiscal 2024?", "7,469"),
    ("What was product revenue in fiscal year 2024?", "64,773"),
    ("How much in short-term investments did Microsoft hold at the end of FY2023?", "76,558"),
    ("Accounts receivable at June 30, 2024", "56,924"),
    ("What was long-term debt on the balance sheet as of June 30, 2024?", "42,688"),
    ("Total liabilities as of June 30, 2023", "205,753"),
    ("Total stockholders' equity at fiscal year-end 2024", "268,477"),
    ("How much inventory did Microsoft have at June 30, 2024?", "1,246"),
    ("Depreciation, amortization, and other in the FY24 cash flow statement", "22,287"),
    ("How much cash was used in investing activities in fiscal 2024?", "96,970"),
    ("Cash paid for acquisitions of companies in fiscal 2024", "69,132"),
    ("Net cash used in financing in FY2023", "43,935"),
    ("How much common stock did Microsoft buy back in fiscal 2022?", "32,696"),
    # Segments and revenue breakdown
    ("Operating income of the Intelligent Cloud segment in FY24", "49,584"),
    ("More Personal Computing revenue in fiscal 2023", "54,734"),
    ("LinkedIn revenue for fiscal 2024", "16,372"),
    ("Search and news advertising revenue FY2024", "12,576"),
    ("How much revenue came from countries outside the United States in fiscal 2024?", "120,418"),
    ("Dynamics products and cloud services revenue FY24", "6,481"),
    # Notes and management's discussion
    ("What was the effective tax rate in fiscal year 2024?", "was 18% and 19%"),
    ("How much did Microsoft pay for Activision Blizzard?", ("$75.4 billion", "75,408")),
    ("How much goodwill was recorded for the Activision Blizzard acquisition?", "$51.0 billion"),
    ("How much was left in the share repurchase program at June 30, 2024?", "$10.3 billion"),
    ("What were total dividends declared in fiscal 2024?", ("$22.3 billion", "22,293")),
    ("What was depreciation expense in fiscal 2024?", "$15.2 billion"),
    ("How fast did Azure and other cloud services revenue grow?", "grew 30%"),
    ("What was total debt as of June 30, 2024?", "44,937"),
    ("Ending unrecognized tax benefits for fiscal 2024", "22,760"),
    ("How much office space does Microsoft own?", "12 million square feet"),
    ("Cash dividends declared per common share in fiscal 2024", "3.00"),
]

# name -> retrieve_and_rerank options
CONFIGURATIONS = {
    "semantic + cross-encoder": {"keyword": False, "reranker": "cross-encoder"},
    "hybrid + cross-encoder": {"keyword": True, "reranker": "cross-encoder"},
    "hybrid + LLM rerank": {"keyword": True, "reranker": "llm"},
}

_FIGURE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\.\d+")


def _evidence(item: dict) -> str | None:
    if item["question"] in EVIDENCE_OVERRIDES:
        return EVIDENCE_OVERRIDES[item["question"]]
    match = _FIGURE.search(item["ground_truth"])
    return match.group(0) if match else None


def _rank(docs: list, evidence: str | tuple[str, ...]) -> int | None:
    wanted = evidence if isinstance(evidence, tuple) else (evidence,)
    for i, doc in enumerate(docs, 1):
        if any(e in doc.page_content for e in wanted):
            return i
    return None


def _score(cases, vectorstore, options: dict) -> list[int | None]:
    return [
        _rank(rag.retrieve_and_rerank(q, vectorstore, PUBLIC_OWNER, **options), ev)
        for q, ev in cases
    ]


def _summary(ranks: list[int | None]) -> str:
    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1)
    hitk = sum(1 for r in ranks if r is not None)
    mrr = sum(1 / r for r in ranks if r) / n
    return f"hit@1 {hit1:>2}/{n}   hit@{rag.TOP_K} {hitk:>2}/{n}   MRR {mrr:.3f}"


def main() -> None:
    config.run_as("system:eval", cache=False)
    configurations = dict(CONFIGURATIONS)
    if "--no-llm" in sys.argv:
        configurations.pop("hybrid + LLM rerank")
    vectorstore = rag._load_vectorstore()
    sets = {
        "evaluate.py questions": [
            (item["question"], ev) for item in EVAL_QUESTIONS if (ev := _evidence(item))
        ],
        "exact-term questions": EXACT_QUESTIONS,
        "more questions": MORE_QUESTIONS,
    }
    totals: dict[str, list[int | None]] = {name: [] for name in configurations}
    for set_name, cases in sets.items():
        print(f"\n{set_name} ({len(cases)})")
        results = {name: _score(cases, vectorstore, opts) for name, opts in configurations.items()}
        for name, ranks in results.items():
            totals[name] += ranks
            print(f"  {name:<26} {_summary(ranks)}")
    print(f"\nall ({sum(len(c) for c in sets.values())})")
    for name, ranks in totals.items():
        print(f"  {name:<26} {_summary(ranks)}")


if __name__ == "__main__":
    main()
