"""
eval_xbrl.py — Benchmark questions generated from the SEC's XBRL data, for any company.

eval_answers.py scores 47 questions checked by hand in two filings. This scores
as many companies as you name, with no hand-checking: the expected answers are
the company's own XBRL values (EDGAR's companyfacts API) for the fiscal year of
its latest 10-K.

For each ticker it:
  1. fetches the latest 10-K from EDGAR and indexes it into a separate store
     (chroma_db/eval_xbrl/), once; later runs reuse it;
  2. asks for reported line items ("What was NVIDIA Corp's net income in fiscal
     2026?") and derived metrics (operating and net margin, free cash flow,
     current ratio, computed with the engine's formulas from XBRL inputs);
  3. scores each answer: the expected figure, formatted the way the engine
     formats it, must be in the answer, from the calculation route, with a
     page citation.

All companies share the store, so naming the company in the question also
tests choosing the right filing. A question is only generated when XBRL has
every value it needs.

With --history N, each ticker's latest N 10-Ks are indexed instead, and the
questions cover every year they report between them: revenue, cost of revenue
and net income year by year, then revenue growth across the whole span. Filings overlap, so
the expected value for a year is what the most recently filed of them reports
(a restatement replacing the first figure), which is how the engine merges a
company's filings.

Cost: indexing a 10-K is about $0.01-0.03 (embeddings and figure extraction),
a question about $0.001. Needs SEC_USER_AGENT and OPENAI_API_KEY in .env.

Usage:
    python -m financial_rag.eval_xbrl                    # the default tickers
    python -m financial_rag.eval_xbrl NVDA KO WMT        # any others
    python -m financial_rag.eval_xbrl --refresh AAPL     # re-index first
    python -m financial_rag.eval_xbrl --history 3 MSFT   # five years from three 10-Ks
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback

from financial_rag import (
    analysis,
    config,
    edgar,
    eval_answers,
    financials,
    ingest,
    rag,
    router,
    xbrl,
)

load_dotenv()

# Six companies with conventional statements, then a bank, an insurer, a REIT and a
# retailer whose 52/53-week year ends in January
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "KO", "JNJ", "JPM", "TRV", "O", "TGT"]
REPORTED = ["revenue", "net_income", "operating_income", "eps_diluted", "total_assets",
            "operating_cash_flow",
            # Banks' line items; asked only of companies that tag them
            "net_interest_income", "noninterest_income", "provision_for_credit_losses",
            "noninterest_expense"]
DERIVED = ["operating_margin", "net_margin", "free_cash_flow", "current_ratio",
           "efficiency_ratio"]
STORE = Path("chroma_db") / "eval_xbrl"
OWNER = "xbrl-eval"
_CITATION = re.compile(r"\.(?:pdf|html?), p\.")


def use_store(path: Path) -> None:
    """Point indexing, retrieval and figures at a store of the benchmark's own."""
    ingest.CHROMA_DIR = path / "chroma"
    rag.CHROMA_DIR = path / "chroma"
    financials.FACTS_DB = path / "figures.sqlite3"


@dataclass(frozen=True)
class Case:
    ticker: str
    metric: str
    fiscal_year: int
    question: str
    expected: str           # as the engine would print it: "$416,161 million", "32.0%"


def company_name(name: str) -> str:
    """EDGAR's "MICROSOFT CORP" as a question would say it: "Microsoft Corp"."""
    return name.title() if name == name.upper() else name


def xbrl_value(facts_json: dict, metric: str, fiscal_year: int,
               accession: str) -> Decimal | None:
    """The value the filing itself tagged, or the only one reported for that year."""
    values = xbrl.annual_values(facts_json, metric, fiscal_year, accession)
    if len(values) == 1:
        return values.pop()
    values = xbrl.annual_values(facts_json, metric, fiscal_year)
    return values.pop() if len(values) == 1 else None


def make_cases(ticker: str, company: str, fiscal_year: int, facts_json: dict,
               accession: str) -> list[Case]:
    who = company_name(company)
    cases = []
    for key in REPORTED:
        value = xbrl_value(facts_json, key, fiscal_year, accession)
        if value is None:
            continue
        item = financials.LINE_ITEMS[key]
        if item.magnitude:
            value = abs(value)
        metric = analysis.METRICS[key]
        cases.append(Case(ticker, key, fiscal_year,
                          f"What was {who}'s {metric.name.lower()} in fiscal {fiscal_year}?",
                          analysis.format_value(value, metric.unit)))
    for key in DERIVED:
        metric = analysis.METRICS[key]
        formula = metric.formulas[0]
        inputs = []
        for name in formula.inputs:
            value = xbrl_value(facts_json, name, fiscal_year, accession)
            if value is None:
                break
            inputs.append(abs(value) if financials.LINE_ITEMS[name].magnitude else value)
        else:
            value = formula.compute(*inputs)
            cases.append(Case(ticker, key, fiscal_year,
                              f"What was {who}'s {metric.name.lower()} in fiscal {fiscal_year}?",
                              analysis.format_value(value, metric.unit)))
    return cases


HISTORY_METRICS = ["revenue", "cost_of_revenue", "net_income"]


def make_history_cases(ticker: str, company: str, years: list[int], facts_json: dict,
                       accessions: set[str]) -> list[Case]:
    """Each year's revenue, cost of revenue and net income, as the latest of the filings
    reports it (Target restated its FY2022-23 cost of sales), then revenue's compound
    growth from the first year to the last."""
    who = company_name(company)
    labels = xbrl.fiscal_year_labels(facts_json)
    cases = []
    expected: dict[tuple[str, int], Decimal] = {}
    for key in HISTORY_METRICS:
        metric = analysis.METRICS[key]
        for year in years:
            value = xbrl.latest_value(facts_json, key, year, accessions, labels)
            if value is None:
                continue
            expected[(key, year)] = value
            cases.append(Case(ticker, key, year,
                              f"What was {who}'s {metric.name.lower()} in fiscal {year}?",
                              analysis.format_value(value, metric.unit)))
    first, last = expected.get(("revenue", years[0])), expected.get(("revenue", years[-1]))
    if first and last and first > 0 and last > 0:
        rate = (last / first) ** (Decimal(1) / (years[-1] - years[0])) - 1
        cases.append(Case(ticker, "revenue_growth", years[-1],
                          f"What was {who}'s compound annual revenue growth from fiscal "
                          f"{years[0]} to fiscal {years[-1]}?",
                          f"{analysis.format_value(rate, 'percent')} a year"))
    return cases


def prepare(ticker: str, refresh: bool = False, history: int = 1) -> tuple[str, list[Case]]:
    """Index the ticker's latest 10-K (or latest `history` 10-Ks) into the store, once,
    and generate its cases."""
    cik = edgar.cik_for_ticker(ticker)
    if cik is None:
        raise SystemExit(f"No SEC filer has the ticker {ticker}.")
    reports = edgar.annual_reports(cik, history)
    if not reports:
        raise SystemExit(f"{ticker} has no 10-K on EDGAR.")
    names = []
    for filing in reports:
        name = f"{ticker.lower()}-10k-{filing.report_date}.htm"
        path = STORE / "filings" / name
        if refresh or not path.exists() or not financials.count_facts(OWNER).get(name):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(edgar.fetch_document(filing.url), encoding="utf-8")
            ingest.ingest_file(path, OWNER, cik=cik)
        names.append(name)
    facts_json = edgar.companyfacts(cik)
    # As the company names it: Target's year to 2026-01-31 is "fiscal 2025"
    labels = xbrl.fiscal_year_labels(facts_json)
    latest = reports[0]
    fiscal_year = xbrl.year_name(date.fromisoformat(latest.report_date), labels)
    if history == 1:
        cases = make_cases(ticker, latest.company, fiscal_year, facts_json, latest.accession)
    else:
        # Each 10-K reports three years of income: n filings cover n + 2 years
        oldest = xbrl.year_name(date.fromisoformat(reports[-1].report_date), labels) - 2
        cases = make_history_cases(ticker, latest.company, list(range(oldest, fiscal_year + 1)),
                                   facts_json, {r.accession for r in reports})
    return ", ".join(names), cases


def run(case: Case) -> dict:
    start = time.perf_counter()
    result = router.answer(case.question, OWNER)
    text = result.answer
    return {
        "ticker": case.ticker, "metric": case.metric, "question": case.question,
        "expected": case.expected, "answer": text, "route": result.route,
        "filing": result.filing,
        "correct": case.expected in text,
        "route_ok": result.route == "calculation",
        "cited": bool(_CITATION.search(text)),
        "seconds": round(time.perf_counter() - start, 2),
    }


def _options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--refresh", action="store_true", help="re-fetch and re-index")
    parser.add_argument("--history", type=int, default=1, metavar="N",
                        help="index each ticker's latest N 10-Ks and ask about every year")


def main(argv: list[str] | None = None) -> int:
    args = eval_answers.benchmark_args(__doc__.split("\n\n")[0].strip(), argv, _options)
    config.run_as("system:eval", cache=False)
    use_store(STORE)
    budget = eval_answers.Budget(args.max_cost)

    def charged(work, *work_args):
        """Run `work`, charging its chat-model calls (figure extraction too) to the budget."""
        with get_usage_metadata_callback() as usage:
            out = work(*work_args)
        budget.charge(eval_answers.usage_cost(usage.usage_metadata)[2])
        return out

    rows = []
    total = 0
    stopped = False
    for ticker in (t.upper() for t in args.tickers):
        if budget.exhausted:
            stopped = True
            break
        name, cases = charged(prepare, ticker, args.refresh, args.history)
        total += len(cases)
        print(f"{ticker}: {name}, {len(cases)} questions", flush=True)
        for case in cases:
            if budget.exhausted:
                stopped = True
                break
            row = charged(run, case)
            rows.append(row)
            mark = "ok  " if row["correct"] and row["route_ok"] else "FAIL"
            print(f"  {mark} {case.metric:<20} expected {case.expected}", flush=True)

    print()
    for ticker in dict.fromkeys(r["ticker"] for r in rows):
        group = [r for r in rows if r["ticker"] == ticker]
        print(f"{ticker:<6} {_summary(group)}")
    print(f"{'all':<6} {_summary(rows)}")
    for r in rows:
        if not (r["correct"] and r["route_ok"] and r["cited"]):
            print(f"\n--- {r['question']}  (expected {r['expected']}, route {r['route']})\n"
                  f"{r['answer'][:500]}")
    out = STORE / "results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nResults written to {out}")

    full = lambda r: r["correct"] and r["route_ok"] and r["cited"]  # noqa: E731
    ok = sum(full(r) for r in rows)
    summary = [f"### XBRL benchmark: {ok}/{total} passed",
               f"estimated chat-model spend ${budget.spent:.3f} (indexing included; "
               "embeddings not counted)", "",
               "| company | questions | correct | route | cited |", "| --- | --- | --- | --- | --- |"]
    for ticker in dict.fromkeys(r["ticker"] for r in rows):
        group = [r for r in rows if r["ticker"] == ticker]
        counts = [sum(r[key] for r in group) for key in ("correct", "route_ok", "cited")]
        summary.append(f"| {ticker} | {len(group)} | " + " | ".join(map(str, counts)) + " |")
    misses = [r for r in rows if not full(r)]
    if misses:
        summary += ["", "Not passed:"] + [f"- {r['question']} (expected {r['expected']})"
                                          for r in misses]
    eval_answers.write_summary(args.summary, summary)
    return eval_answers.finish(ok, total, stopped, budget, args.min_score, "eval_xbrl")


def _summary(rows: list[dict]) -> str:
    n = len(rows)
    count = lambda key: sum(1 for r in rows if r[key])  # noqa: E731
    return f"correct {count('correct'):>3}/{n}   route {count('route_ok'):>3}/{n}   " \
           f"cited {count('cited'):>3}/{n}"


if __name__ == "__main__":
    raise SystemExit(main())
