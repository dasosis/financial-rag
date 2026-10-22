"""
financials.py — Structured financial records extracted from 10-K statements.

Text retrieval is the wrong tool for "what was the operating margin?": the
answer is a calculation over specific line items. This module turns the three
primary statements (income, balance sheet, cash flows) into typed records that
analysis.py computes on.

Extraction is split so that no number ever comes from a language model:

  1. Code picks the statement pages (heading near the top, dense with figures).
  2. The model maps each canonical line item to the row that reports it: the
     page, the row label as printed, and one sample value with its fiscal year.
  3. Code verifies the mapping against the page text: the row must exist, its
     label must match, and the sample value must sit in the column whose header
     year matches. Only then are the values read, for every year column, from
     the PDF text itself and scaled by the page's "(In millions ...)" header.

Every stored figure keeps the file, page and row it was read from, so any
calculation can cite each of its inputs.

Breakdowns of the totals are extracted the same way: revenue and operating
income by reportable segment, revenue by geography, and revenue by product or
service line, from the notes. A breakdown is stored only if its parts add up
exactly to the verified company total for every year, so a mis-mapped table
(say, long-lived assets by country) can't pass for revenue.

Records live in a SQLite file inside the vector store directory, so rebuilding
the store (start.py --reingest) rebuilds both together.

Usage:
    python -m financial_rag.financials              # (re)extract figures for every PDF in data/
    python -m financial_rag.financials FILE.pdf     # print the figures extracted from one PDF
"""

from __future__ import annotations

import inspect
import re
import sqlite3
import sys
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from financial_rag import config
from financial_rag.tracing import note, traced

load_dotenv()

FACTS_DB = Path("chroma_db") / "financials.sqlite3"
MAX_STATEMENT_PAGES = 8


# ---------------------------------------------------------------------------
# Canonical line items
# ---------------------------------------------------------------------------


def page_number(index) -> int | str:
    """
    The page number a reader sees for a stored page index.

    Chunks and figures store PyPDF's 0-based page index; PDF viewers and the
    pages' own printed footers ("67/139") count from 1. Every citation shows
    this number, so "p.67" opens the right page. Unknown pages show as "?".
    """
    try:
        return int(index) + 1
    except (TypeError, ValueError):
        return "?"

@dataclass(frozen=True)
class LineItem:
    key: str
    name: str
    statement: str      # income | balance | cash
    unit: str           # USD | USD/share | shares
    magnitude: bool     # store the absolute value: outflows print in parentheses
    hint: str           # guidance for mapping the row; never used for values


LINE_ITEMS: dict[str, LineItem] = {
    item.key: item
    for item in [
        LineItem("revenue", "Total revenue", "income", "USD", False,
                 "Total revenue / total net sales / net revenues (the total row, not a segment)."),
        LineItem("cost_of_revenue", "Cost of revenue", "income", "USD", True,
                 "Total cost of revenue / cost of sales (the total row)."),
        LineItem("gross_profit", "Gross profit", "income", "USD", False,
                 "Gross profit, or the dollar row labelled 'Gross margin'. Only if printed."),
        LineItem("research_and_development", "Research and development", "income", "USD", True,
                 "Research and development expense."),
        # Banks: no cost of revenue; revenue is net interest income plus noninterest revenue
        LineItem("net_interest_income", "Net interest income", "income", "USD", False,
                 "Banks: net interest income (interest income minus interest expense), "
                 "before the provision for credit losses."),
        LineItem("noninterest_income", "Noninterest revenue", "income", "USD", False,
                 "Banks: total noninterest revenue / noninterest income (fees, trading and "
                 "other non-interest revenue)."),
        LineItem("provision_for_credit_losses", "Provision for credit losses", "income", "USD",
                 False, "Banks: provision for credit losses / provision for loan losses."),
        LineItem("noninterest_expense", "Noninterest expense", "income", "USD", True,
                 "Banks: total noninterest expense (the total row)."),
        LineItem("operating_income", "Operating income", "income", "USD", False,
                 "Operating income / income from operations / operating loss."),
        LineItem("income_before_taxes", "Income before income taxes", "income", "USD", False,
                 "Income before income taxes / pre-tax income."),
        LineItem("income_tax_expense", "Provision for income taxes", "income", "USD", False,
                 "Provision for (benefit from) income taxes / income tax expense."),
        LineItem("net_income", "Net income", "income", "USD", False,
                 "Net income (loss) attributable to the company's shareholders, after "
                 "noncontrolling interests: the line some filings print below 'consolidated "
                 "net income'. Also called net earnings."),
        LineItem("eps_basic", "Basic earnings per share", "income", "USD/share", False,
                 "Basic earnings (loss) per share."),
        LineItem("eps_diluted", "Diluted earnings per share", "income", "USD/share", False,
                 "Diluted earnings (loss) per share."),
        LineItem("diluted_shares", "Diluted weighted average shares", "income", "shares", False,
                 "Diluted weighted average shares outstanding."),
        LineItem("cash_and_equivalents", "Cash and cash equivalents", "balance", "USD", False,
                 "Cash and cash equivalents on the balance sheet (not including investments)."),
        LineItem("total_current_assets", "Total current assets", "balance", "USD", False,
                 "Total current assets."),
        LineItem("total_assets", "Total assets", "balance", "USD", False,
                 "Total assets."),
        LineItem("total_current_liabilities", "Total current liabilities", "balance", "USD", False,
                 "Total current liabilities."),
        LineItem("total_liabilities", "Total liabilities", "balance", "USD", False,
                 "Total liabilities. Only if printed as its own row."),
        LineItem("long_term_debt", "Long-term debt", "balance", "USD", False,
                 "Long-term debt, the non-current row (not the current portion)."),
        LineItem("total_equity", "Total stockholders' equity", "balance", "USD", False,
                 "Total stockholders'/shareholders' equity (not 'total liabilities and equity')."),
        LineItem("operating_cash_flow", "Net cash from operating activities", "cash", "USD", False,
                 "Net cash provided by (used in) operating activities / net cash from operations."),
        LineItem("capital_expenditures", "Capital expenditures", "cash", "USD", True,
                 "Capital expenditures / purchases of or additions to property and equipment."),
        LineItem("dividends_paid", "Dividends paid", "cash", "USD", True,
                 "Cash dividends paid to common stockholders."),
        LineItem("share_repurchases", "Share repurchases", "cash", "USD", True,
                 "Common stock repurchased / repurchases of common stock."),
    ]
}

LineItemKey = Literal[tuple(LINE_ITEMS)]  # type: ignore[valid-type]

# What a row reporting each line item is called, in some wording: the printed
# label must fit. Asked for line items a statement doesn't have, the model
# substitutes a neighbour despite being told not to (a bank's "Total noninterest
# expense" for cost of revenue, an insurer's "Total assets" for total current
# assets, Coca-Cola's "Other operating charges" for R&D), and XBRL can only catch
# that when the company tags the line item. Checked against every row XBRL
# confirmed for ten companies.
PRINTED_AS: dict[str, re.Pattern] = {
    key: re.compile(pattern, re.IGNORECASE)
    for key, pattern in {
        "revenue": r"revenue|\bsales\b",
        "cost_of_revenue": r"\bcosts? of\b",
        "gross_profit": r"\bgross (?:profit|margin)",
        "research_and_development": r"\bresearch\b",
        "net_interest_income": r"\bnet interest income\b",
        "noninterest_income": r"\bnon-?interest (?:revenue|income)\b",
        "provision_for_credit_losses": r"\bprovision\b.*\b(?:credit|loan) loss",
        "noninterest_expense": r"\bnon-?interest expense",
        "operating_income": r"\boperat",
        "income_before_taxes": r"\bbefore\b.*\btax|\bpre-?tax",
        "income_tax_expense": r"\btax",
        "net_income": r"\bnet\b.{0,8}\b(?:income|earnings|loss)",
        "eps_basic": r"\bbasic\b",
        "eps_diluted": r"\bdiluted\b",
        "diluted_shares": r"\bdiluted\b",
        "cash_and_equivalents": r"\bcash\b",
        "total_current_assets": r"\bcurrent assets\b",
        "total_assets": r"\btotal assets\b",
        "total_current_liabilities": r"\bcurrent liabilities\b",
        "total_liabilities": r"\btotal liabilities\b(?! and)",
        "long_term_debt": r"\bdebt\b|\bborrowings\b|\bnotes payable\b",
        "total_equity": r"^(?!.*\bliabilities\b).*\b(?:equity|investment|deficit)\b",
        "operating_cash_flow": r"\boperat",
        "capital_expenditures":
            r"\bproperty\b|\bequipment\b|\bcapital expenditures?\b|\bproductive assets\b",
        "dividends_paid": r"\bdividends?\b|\bdistributions?\b",
        "share_repurchases":
            r"\brepurchase|\btreasury\b|\bbuybacks?\b|\bpurchases? of (?:common )?stock\b",
    }.items()
}


@dataclass(frozen=True)
class Breakdown:
    """A total split into parts: a segment, a geographic area, a product line."""
    key: str
    name: str
    dimension: str      # segment | geography | product
    total: str          # the LINE_ITEMS key the parts must add up to


BREAKDOWNS: dict[str, Breakdown] = {
    b.key: b
    for b in [
        Breakdown("segment_revenue", "Segment revenue", "segment", "revenue"),
        Breakdown("segment_operating_income", "Segment operating income", "segment",
                  "operating_income"),
        Breakdown("geographic_revenue", "Revenue by geography", "geography", "revenue"),
        Breakdown("product_revenue", "Revenue by product or service", "product", "revenue"),
    ]
}

BreakdownKey = Literal[tuple(BREAKDOWNS)]  # type: ignore[valid-type]

# Display names for every stored metric key
ITEM_NAMES: dict[str, str] = {
    **{k: item.name for k, item in LINE_ITEMS.items()},
    **{k: b.name for k, b in BREAKDOWNS.items()},
}


@dataclass(frozen=True)
class Fact:
    """One verified figure: a line item's value for one fiscal year."""

    owner: str
    source_file: str
    metric: str
    fiscal_year: int
    value: Decimal      # normalised: dollars, dollars per share, or shares
    unit: str
    label: str          # row label as printed in the filing
    page: int
    line: str           # the row as printed: the citation excerpt
    segment: str = ""   # for breakdowns: the part ("Intelligent Cloud"); "" for totals
    xbrl: str = ""      # "match" when the SEC's XBRL data reports the same value


# ---------------------------------------------------------------------------
# Reading numbers from statement text
# ---------------------------------------------------------------------------

_DASHES = {"—", "–", "-"}   # some filings print zero as a dash
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_GROUPED = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
# "(In millions, except ...)", or the bare "(millions, except per share data)" Target prints
_SCALE = re.compile(
    r"\bin\s+(thousands|millions|billions)\b|\(\s*(thousands|millions|billions)\s*[,)]",
    re.IGNORECASE)
# Within a scale note: shares with their own scale ("shares ... in thousands"), or
# shares excepted from it ("except share data"). "per share" amounts are neither.
_SHARE_SCALE = re.compile(
    r"(?<!per )\bshares?\b[^)]*?\bin\s+(thousands|millions|billions)\b", re.IGNORECASE)
_SHARES_UNSCALED = re.compile(r"\bexcept\b[^)]*?(?<!per )\bshares?\b", re.IGNORECASE)
# "September 27," / "June 30" / "Dec. 31,": the date above a year in a column header
_DATE_FRAGMENT = re.compile(r"[A-Z][a-z]{2,8}\.?\s+\d{1,2},?")
_SCALES = {"thousands": Decimal(1_000), "millions": Decimal(1_000_000),
           "billions": Decimal(1_000_000_000)}
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"a", "and", "of", "the", "for", "in", "to"}


def parse_number(token: str) -> Decimal | None:
    """Parse one printed figure: '245,122', '$ 11.80', '(1,646)', '-'. None if not a number."""
    s = token.strip().replace("$", "").replace(" ", "")
    if s in _DASHES:
        return Decimal(0)
    negative = (s.startswith("(") and s.endswith(")")) or s.startswith("-")
    s = s.strip("()").lstrip("-")
    if not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", s):
        return None
    try:
        value = Decimal(s.replace(",", ""))
    except InvalidOperation:
        return None
    return -value if negative else value


def split_row(line: str) -> tuple[str, list[Decimal]]:
    """Split a statement row into its label and the figures at the end of the line.

    Figures are read right to left until the first non-number, so numbers inside
    the label ("allowance of $830 and $650") only join the run if they touch it;
    callers take just as many trailing figures as there are year columns.
    """
    s = re.sub(r"\(\s+", "(", line)
    s = re.sub(r"\s+\)", ")", s)
    tokens = s.split()
    values: list[Decimal] = []
    i = len(tokens)
    while i > 0:
        token = tokens[i - 1]
        if token == "$":
            i -= 1
            continue
        value = parse_number(token)
        if value is None:
            break
        values.append(value)
        i -= 1
    values.reverse()
    label = " ".join(t for t in tokens[:i] if t != "$")
    return label, values


def header_years(lines: list[str], row: int) -> list[int] | None:
    """
    Column years from the nearest header above `row`: one line ('... 2024 2023 2022'),
    or one year per line, as some PDFs extract a header of dated columns
    ('September 27,' / '2025' / 'September 28,' / '2024').
    """
    stacked: list[int] = []
    for j in range(row - 1, -1, -1):
        line = lines[j].strip()
        if _GROUPED.search(line):
            if stacked:
                break
            continue    # a data row, not a header
        if _YEAR.fullmatch(line):
            stacked.insert(0, int(line))
            continue
        if stacked:
            if _DATE_FRAGMENT.fullmatch(line):
                continue
            break
        years = [int(y) for y in _YEAR.findall(line)]
        if len(years) >= 2 and len(set(years)) == len(years):
            return years
    if len(stacked) >= 2 and len(set(stacked)) == len(stacked):
        return stacked
    return None


def page_scale(text: str, unit: str = "USD") -> Decimal | None:
    """
    Multiplier from the statement's '(In millions, except per share amounts)' note.

    Share counts can be scaled differently from dollars: '(In millions, except
    number of shares, which are reflected in thousands)' puts them in thousands,
    and '(In thousands, except share data)' leaves them unscaled.
    """
    match = _SCALE.search(text)
    if not match:
        return None
    if unit == "shares":
        end = text.find(")", match.start())
        note = text[match.start(): end if end != -1 else match.end() + 200]
        if own := _SHARE_SCALE.search(note):
            return _SCALES[own.group(1).lower()]
        if _SHARES_UNSCALED.search(note):
            return Decimal(1)
    return _SCALES[(match.group(1) or match.group(2)).lower()]


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOPWORDS


def same_words(a: str, b: str) -> bool:
    """True if two labels differ only in case, punctuation or mis-encoded characters."""
    return _words(a) == _words(b)


def label_score(expected: str, printed: str) -> float:
    """Share of the expected label's words that appear in the printed label."""
    want = _words(expected)
    return len(want & _words(printed)) / len(want) if want else 0.0


LABEL_THRESHOLD = 0.75


@dataclass(frozen=True)
class VerifiedRow:
    label: str
    excerpt: str
    years: list[int]
    values: list[Decimal]


def verify_row(page_text: str, label: str, value_text: str, fiscal_year: int) -> VerifiedRow | None:
    """
    Find the row the model pointed at and confirm it, or return None.

    The row must carry `value_text` in the column whose header year is
    `fiscal_year`, and its label (allowing two wrapped lines above) must match
    `label`. If two different rows qualify equally, the mapping is ambiguous and
    rejected rather than guessed.
    """
    target = parse_number(value_text)
    if target is None:
        return None
    label = _FOOTNOTE.sub("", label)    # the model may copy a footnote marker
    lines = [ln.rstrip() for ln in page_text.splitlines()]
    candidates: list[tuple[float, VerifiedRow]] = []
    for i, line in enumerate(lines):
        row_label, values = split_row(line)
        if not values:
            continue
        years = header_years(lines, i)
        if not years or len(values) < len(years) or fiscal_year not in years:
            continue
        columns = values[-len(years):]
        if abs(columns[years.index(fiscal_year)]) != abs(target):
            continue
        # Labels can wrap: try the row alone, then with up to two lines above.
        best_score, best_label, best_excerpt = -1.0, row_label, line.strip()
        parts = [row_label]
        excerpt = [line.strip()]
        for k in range(i, max(i - 3, -1), -1):
            if k < i:
                above_label, above_values = split_row(lines[k])
                if above_values:
                    break   # the line above is another data row
                parts.insert(0, above_label)
                excerpt.insert(0, lines[k].strip())
            joined = " ".join(p for p in parts if p)
            score = label_score(label, joined)
            if score > best_score:
                best_score, best_label, best_excerpt = score, joined, " ".join(excerpt)
        if best_score >= LABEL_THRESHOLD:
            candidates.append(
                (best_score, VerifiedRow(best_label, best_excerpt, years, columns))
            )
    if not candidates:
        return None
    top = max(score for score, _ in candidates)
    best = [row for score, row in candidates if score == top]
    if len(best) > 1:
        # "Net earnings" matches both "Net earnings" and "Net earnings from
        # continuing operations" fully; the row labelled exactly as asked wins
        exact = [row for row in best if same_words(label, row.label)]
        if exact:
            best = exact
    if any(row.values != best[0].values for row in best[1:]):
        return None     # equally good rows disagree: ambiguous
    return best[0]


# ---------------------------------------------------------------------------
# Choosing the statement pages
# ---------------------------------------------------------------------------

_HEADINGS = {
    "income": re.compile(
        r"income\s+statements?|statements?\s+of\s+(?:consolidated\s+)?(?:operations|income|earnings)",
        re.IGNORECASE),
    "balance": re.compile(
        r"balance\s+sheets?|statements?\s+of\s+financial\s+(?:position|condition)", re.IGNORECASE),
    "cash": re.compile(
        r"cash\s+flows?\s+statements?|statements?\s+of\s+(?:consolidated\s+)?cash\s+flows?",
        re.IGNORECASE),
}
# "Comprehensive income", "comprehensive (loss) income", "comprehensive loss"
_COMPREHENSIVE = re.compile(r"comprehensive(?:\s+\(?(?:income|loss|earnings)\)?)+",
                            re.IGNORECASE)
_HEADING_LINES = 8          # the heading must sit near the top of the page
_MIN_FIGURES = 12           # comma-grouped numbers; a TOC page has almost none


def statement_pages(pages: dict[int, str]) -> dict[int, str]:
    """
    Pages that hold a primary financial statement, by page number.

    A page qualifies when a statement heading appears in its first few lines
    (which excludes notes that merely mention "balance sheets") and it is dense
    with figures (which excludes the table of contents).

    Notes and MD&A tables can still qualify (Coca-Cola's had five), and they
    crowd the row mapping. The primary statements sit together, so when the
    income statement, balance sheet and cash flow statement are all found, only
    the tightest run of pages holding one of each is kept. Pages that match all
    three headings at once (a note listing them) don't anchor that run.
    """
    chosen: dict[int, str] = {}
    kinds: dict[int, set[str]] = {}
    for number in sorted(pages):
        text = pages[number]
        top = [ln for ln in text.splitlines() if ln.strip()][:_HEADING_LINES]
        # A row with figures isn't a heading, even one that names a statement:
        # Travelers' comprehensive income lists gains "recognized in the
        # consolidated statement of income 2,744 (644) 2,249". A statement of
        # comprehensive income alone isn't the income statement, but Realty
        # Income's "Statements of Income and Comprehensive Income" is.
        heading = "\n".join(_COMPREHENSIVE.sub("", ln) for ln in top if not _GROUPED.search(ln))
        if len(_GROUPED.findall(text)) < _MIN_FIGURES:
            continue
        found = {kind for kind, rx in _HEADINGS.items() if rx.search(heading)}
        if found:
            chosen[number] = text
            kinds[number] = found
    window = _statement_window({n: k.pop() for n, k in kinds.items() if len(k) == 1})
    if window:
        low, high = window
        chosen = {n: t for n, t in chosen.items() if low <= n <= high}
    return dict(list(chosen.items())[:MAX_STATEMENT_PAGES])


def _statement_window(kind_of: dict[int, str]) -> tuple[int, int] | None:
    """The shortest page range holding all three statements, earliest on a tie."""
    best = None
    numbers = sorted(kind_of)
    for i, low in enumerate(numbers):
        seen: set[str] = set()
        for high in numbers[i:]:
            seen.add(kind_of[high])
            if len(seen) == len(_HEADINGS):
                if best is None or high - low < best[1] - best[0]:
                    best = (low, high)
                break
    return best


# ---------------------------------------------------------------------------
# Row mapping (the model) and extraction (code)
# ---------------------------------------------------------------------------

class RowMapping(BaseModel):
    metric: LineItemKey
    page: int
    label: str
    value: str
    fiscal_year: int


class RowMappings(BaseModel):
    rows: list[RowMapping]


_MAPPING_PROMPT = """You map line items in a company's financial statements to the rows \
that report them. You will get the text of the statement pages, each starting with \
"=== Page N ===".

For each line item below that the statements print as its own row, return:
- metric: the line item key
- page: the page number N the row is on
- label: the row label exactly as printed
- value: one value from that row exactly as printed (e.g. "245,122" or "(44,477 )")
- fiscal_year: the fiscal year of the column that value is in

Rules:
- Use rows from the primary statements only. Never compute, add up or derive a value.
- Skip a line item if no single row reports it. Do not substitute a different row.
- Each metric at most once.

Line items:
{items}
"""


def _map_rows(pages: dict[int, str], only: list[str] | None = None) -> list[RowMapping]:
    """Ask the model which row reports each line item (or only those in `only`).

    Its values are only hints, checked by verify_row.
    """
    items = "\n".join(f"- {item.key}: {item.hint}" for item in LINE_ITEMS.values()
                      if only is None or item.key in only)
    text = "\n\n".join(f"=== Page {n} ===\n{t}" for n, t in pages.items())
    llm = config.chat_model(temperature=0).with_structured_output(RowMappings)
    result = llm.invoke(
        [("system", _MAPPING_PROMPT.format(items=items)), ("human", text)]
    )
    return result.rows


def extract_facts(
    pages: dict[int, str], source_file: str, owner: str, mapper=_map_rows
) -> list[Fact]:
    """
    Extract verified figures from a filing's pages ({page number: text}).

    Mappings that fail verification are dropped. If two mappings for the same
    metric produce different values for a year, that metric is dropped entirely
    rather than picking one.

    Asked for all 22 line items at once, the model sometimes leaves a few out
    (Coca-Cola's capital expenditures in two runs of three). So a mapper that
    takes `only` is asked a second time, for just the items still without a
    verified figure; its answers are verified the same way.
    """
    statements = statement_pages(pages)
    if not statements:
        print(f"No financial statement pages found in {source_file}.")
        return []

    found: dict[tuple[str, int], Fact] = {}
    conflicted: set[str] = set()
    rejected = 0
    mappings = list(mapper(statements))
    second_pass = "only" in inspect.signature(mapper).parameters
    while mappings or second_pass:
        if not mappings:
            # First pass done: ask again for items still without a verified figure
            second_pass = False
            missing = [key for key in LINE_ITEMS if key not in {m for m, _ in found}]
            mappings = list(mapper(statements, only=missing)) if missing else []
            continue
        mapping = mappings.pop(0)
        item = LINE_ITEMS.get(mapping.metric)
        text = statements.get(mapping.page)
        if item is None or text is None:
            rejected += 1
            continue
        row = verify_row(text, mapping.label, mapping.value, mapping.fiscal_year)
        if row is None or not PRINTED_AS[item.key].search(row.label):
            rejected += 1
            continue
        if item.unit == "USD/share":
            multiplier = Decimal(1)
        else:
            multiplier = page_scale(text, item.unit)
            if multiplier is None:
                rejected += 1   # no "(In millions ...)" note: scale unknown
                continue
        for year, raw in zip(row.years, row.values, strict=True):
            value = (abs(raw) if item.magnitude else raw) * multiplier
            fact = Fact(owner, source_file, item.key, year, value, item.unit,
                        row.label, mapping.page, row.excerpt)
            previous = found.get((item.key, year))
            if previous is not None and previous.value != value:
                conflicted.add(item.key)
            found.setdefault((item.key, year), fact)

    facts = align_balance_years(
        [f for (metric, _), f in sorted(found.items()) if metric not in conflicted])
    print(
        f"Extracted {len(facts)} figures from {source_file} "
        f"({rejected} unverified mappings dropped"
        + (f", conflicting rows for {sorted(conflicted)}" if conflicted else "")
        + ")."
    )
    return facts


def align_balance_years(facts: list[Fact]) -> list[Fact]:
    """
    Name balance sheet columns by the fiscal year they close, as the income statement does.

    Most companies name a fiscal year after the calendar year it ends in, so the
    statements' headers agree. Some name it after the year it mostly falls in:
    Target heads its income statement for the year to January 31, 2026 "2025",
    while its balance sheet is dated "January 31, 2026". A balance sheet is never
    dated after the latest period its own income statement covers, so when its
    latest year is one more, each of its columns closes the year named one less.
    """
    flows = [f.fiscal_year for f in facts if LINE_ITEMS[f.metric].statement != "balance"]
    balances = [f.fiscal_year for f in facts if LINE_ITEMS[f.metric].statement == "balance"]
    if not flows or not balances or max(balances) != max(flows) + 1:
        return facts
    return [replace(f, fiscal_year=f.fiscal_year - 1)
            if LINE_ITEMS[f.metric].statement == "balance" else f
            for f in facts]


def row_printing(
    pages: dict[int, str], metric: str, expected: dict[int, set[Decimal]],
    owner: str, source_file: str,
) -> list[Fact]:
    """
    The figures of the one statement row that prints the given values, or [].

    `expected` holds a line item's values by fiscal year, as XBRL reports them.
    When XBRL contradicts the row the model chose (Realty Income's "Net income"
    before noncontrolling interests, where XBRL reports "Net income attributable
    to the Company"), the right row is usually on the same pages. A row qualifies
    if its label fits the line item and, in every year XBRL reports, it prints
    XBRL's value. Statements that print an expense as a deduction ("Income taxes
    (85,346)") differ from XBRL only in sign; XBRL's sign is kept.

    Balance sheet columns may be dated a year after the year they close (see
    align_balance_years), so both namings are tried. If rows with different
    values qualify, none is chosen.
    """
    item = LINE_ITEMS[metric]
    shifts = (0, -1) if item.statement == "balance" else (0,)
    found: dict[tuple[Decimal, ...], tuple[int, str, str, list[int], Decimal]] = {}
    for number, text in pages.items():
        scale = Decimal(1) if item.unit == "USD/share" else page_scale(text, item.unit)
        if scale is None:
            continue
        lines = [ln.rstrip() for ln in text.splitlines()]
        for i, line in enumerate(lines):
            label, values = split_row(line)
            if not values or not PRINTED_AS[metric].search(label):
                continue
            printed_years = header_years(lines, i)
            if not printed_years or len(values) < len(printed_years):
                continue
            columns = [v * scale for v in values[-len(printed_years):]]
            for shift in shifts:
                by_year = dict(zip((y + shift for y in printed_years), columns, strict=True))
                checked = {y: v for y, v in by_year.items() if expected.get(y)}
                if checked and all(abs(v) in {abs(x) for x in expected[y]}
                                   for y, v in checked.items()):
                    found.setdefault(tuple(columns), (number, label, line.strip(), by_year))
    if len(found) != 1:
        return []
    [(page, label, excerpt, by_year)] = found.values()

    # XBRL's sign, where only the sign differs: in every year it reports, or none
    flipped = [v not in expected[y] for y, v in by_year.items() if expected.get(y) and v]
    if any(flipped) and not all(flipped):
        return []
    sign = -1 if any(flipped) else 1
    return [
        Fact(owner, source_file, metric, year, abs(v) if item.magnitude else v * sign,
             item.unit, label, page, excerpt, xbrl="match" if expected.get(year) else "")
        for year, v in by_year.items()
    ]


# ---------------------------------------------------------------------------
# Breakdowns: segments, geography, product lines
# ---------------------------------------------------------------------------

_BREAKDOWN_PAGE = re.compile(
    r"segment|geographic|product and service|by product|offerings|\bcountries\b",
    re.IGNORECASE)
MAX_BREAKDOWN_PAGES = 6
_NUMBERS_ONLY = re.compile(r"^[\s$]*(?:\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)?[\s$]*)+$")
_FOOTNOTE = re.compile(r"\s*\([a-z0-9]\)\s*$")     # "Services (1)", "Other (a)"


def rejoin_rows(text: str) -> str:
    """
    Put back rows that PDF text extraction split across lines, e.g.
    "Gaming 21,503" / "" / "15,466" / "" / "16,230" -> "Gaming 21,503 15,466 16,230".
    Number-only lines are appended to the data row above them.
    """
    out: list[str] = []
    pending_blank = 0
    for line in text.splitlines():
        if not line.strip():
            pending_blank += 1
            continue
        years_only = all(_YEAR.fullmatch(t) for t in line.split())
        if (_NUMBERS_ONLY.match(line) and not years_only and out
                and split_row(out[-1])[1] and split_row(out[-1])[0]):
            out[-1] = f"{out[-1].rstrip()}  {line.strip()}"
        else:
            out.extend([""] * pending_blank)
            out.append(line)
        pending_blank = 0
    return "\n".join(out)


def breakdown_pages(pages: dict[int, str]) -> dict[int, str]:
    """Note and MD&A pages that may hold breakdown tables, densest first, rejoined."""
    candidates = [
        (len(_GROUPED.findall(text)), number, text)
        for number, text in pages.items()
        if _BREAKDOWN_PAGE.search(text) and len(_GROUPED.findall(text)) >= 6
    ]
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return {number: rejoin_rows(text) for _, number, text in candidates[:MAX_BREAKDOWN_PAGES]}


class PartMapping(BaseModel):
    breakdown: BreakdownKey
    part: str
    page: int
    label: str
    value: str
    fiscal_year: int


class PartMappings(BaseModel):
    rows: list[PartMapping]


_BREAKDOWN_PROMPT = """You find tables in a company's 10-K that split a company total into \
its parts, and map each part's row. You will get the text of some pages, each starting \
with "=== Page N ===".

The breakdowns to look for:
{breakdowns}

For every row that is one part of such a table (never the "Total" row), return:
- breakdown: the breakdown key
- part: the part's name, e.g. "Intelligent Cloud", "United States", "LinkedIn"
- page: the page number N the row is on
- label: the row label exactly as printed
- value: one value from that row exactly as printed
- fiscal_year: the fiscal year of the column that value is in

Rules:
- Only tables whose parts add up to the company's total revenue or total operating income \
for the year. Skip other tables that list segments or countries (assets, goodwill, \
headcount, unearned revenue, ...).
- Never compute, add up or derive a value.
- Each (breakdown, part) at most once.
"""


def _map_parts(pages: dict[int, str]) -> list[PartMapping]:
    """Ask the model which rows are the parts of each breakdown. Values are only hints."""
    breakdowns = "\n".join(
        f"- {b.key}: {ITEM_NAMES[b.total].lower()} by {b.dimension}" for b in BREAKDOWNS.values()
    )
    text = "\n\n".join(f"=== Page {n} ===\n{t}" for n, t in pages.items())
    llm = config.chat_model(temperature=0).with_structured_output(PartMappings)
    return llm.invoke(
        [("system", _BREAKDOWN_PROMPT.format(breakdowns=breakdowns)), ("human", text)]
    ).rows


_Parts = dict[str, tuple["VerifiedRow", int, Decimal]]
_TOTAL_LABEL = re.compile(r"^\s*total\b", re.IGNORECASE)


def _part_sum(members: _Parts, year: int) -> Decimal:
    return sum((row.values[row.years.index(year)] * scale
                for row, _, scale in members.values()), Decimal(0))


def _adds_up(members: _Parts, years: set[int], metric: str,
             total: dict[tuple[str, int], Decimal]) -> bool:
    return bool(years) and len(members) >= 2 and all(
        (metric, year) in total and _part_sum(members, year) == total[(metric, year)]
        for year in years
    )


def _missing_part(
    members: _Parts, years: set[int], metric: str,
    total: dict[tuple[str, int], Decimal], pages: dict[int, str],
) -> tuple[str, tuple[VerifiedRow, int, Decimal]] | None:
    """The one unmapped row on the parts' pages that fills the gap to the total.

    The model sometimes skips a small row (Microsoft's "Other", $45 million of
    $245 billion). If exactly one row reports the shortfall in every year, it is
    the missing part; if none or several do, nothing is added. An overshoot
    (a wrong row mapped) leaves a negative gap and is never filled.
    """
    if not years or any((metric, y) not in total for y in years):
        return None
    gap = {y: total[(metric, y)] - _part_sum(members, y) for y in years}
    if any(g <= 0 for g in gap.values()):
        return None
    taken = {row.excerpt for row, _, _ in members.values()}
    found = []
    for page in sorted({page for _, page, _ in members.values()}):
        text = pages.get(page, "")
        scale = page_scale(text)
        lines = [ln.rstrip() for ln in text.splitlines()]
        for i, line in enumerate(lines):
            label, values = split_row(line)
            if not label or _TOTAL_LABEL.match(label) or line.strip() in taken or not scale:
                continue
            header = header_years(lines, i)
            if not header or len(values) < len(header) or not years <= set(header):
                continue
            columns = values[-len(header):]
            if all(columns[header.index(y)] * scale == gap[y] for y in years):
                name = _FOOTNOTE.sub("", label).strip()
                found.append((name, (VerifiedRow(label, line.strip(), header, columns),
                                     page, scale)))
    return found[0] if len(found) == 1 else None


def extract_breakdowns(
    pages: dict[int, str], source_file: str, owner: str, totals: list[Fact], mapper=_map_parts
) -> list[Fact]:
    """
    Verified parts of the company totals in `totals` (revenue, operating income).

    Each mapped row is verified like a statement row. Then each breakdown is
    kept only if its parts, for every year they report, add up exactly to the
    company total for that year. Anything else (a missing part, a row from
    another table, a wrong year) fails that check, and the whole breakdown is
    dropped rather than stored incomplete.
    """
    candidates = breakdown_pages(pages)
    if not candidates:
        return []
    total = {(f.metric, f.fiscal_year): f.value for f in totals}
    parts: dict[str, dict[str, tuple[VerifiedRow, int, Decimal]]] = {}
    for mapping in mapper(candidates):
        text = candidates.get(mapping.page)
        breakdown = BREAKDOWNS.get(mapping.breakdown)
        if text is None or breakdown is None:
            continue
        row = verify_row(text, mapping.label, mapping.value, mapping.fiscal_year)
        scale = page_scale(text)
        if row is None or scale is None:
            continue
        name = _FOOTNOTE.sub("", row.label).strip()
        parts.setdefault(breakdown.key, {}).setdefault(name, (row, mapping.page, scale))

    facts: list[Fact] = []
    for key, members in parts.items():
        breakdown = BREAKDOWNS[key]
        years = set.intersection(*(set(row.years) for row, _, _ in members.values()))
        if not _adds_up(members, years, breakdown.total, total):
            missing = _missing_part(members, years, breakdown.total, total, candidates)
            if missing is not None:
                members[missing[0]] = missing[1]
                print(f"Completed {key} from {source_file} with {missing[0]!r}, the one row "
                      "on the page that makes its parts add up.")
        if not _adds_up(members, years, breakdown.total, total):
            print(f"Dropped {key} from {source_file}: its parts don't add up to the total.")
            continue
        for name, (row, page, scale) in members.items():
            for year in sorted(years):
                facts.append(Fact(owner, source_file, key, year,
                                  row.values[row.years.index(year)] * scale, "USD",
                                  row.label, page, row.excerpt, name))
    if facts:
        kinds = sorted({f.metric for f in facts})
        print(f"Extracted {len(facts)} breakdown figures from {source_file} ({', '.join(kinds)}).")
    return facts


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS figures (
    owner TEXT NOT NULL,
    source_file TEXT NOT NULL,
    metric TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    value TEXT NOT NULL,
    unit TEXT NOT NULL,
    label TEXT NOT NULL,
    page INTEGER NOT NULL,
    line TEXT NOT NULL,
    segment TEXT NOT NULL DEFAULT '',
    xbrl TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (owner, source_file, metric, segment, fiscal_year)
)
"""


def _connect() -> sqlite3.Connection:
    FACTS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FACTS_DB)
    conn.execute(_SCHEMA)
    # Databases from before breakdowns have a "facts" table keyed without a
    # segment: carry its rows over once, as company totals.
    legacy = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'facts'"
    ).fetchone()
    if legacy:
        with conn:
            conn.execute("INSERT OR IGNORE INTO figures SELECT *, '', '' FROM facts")
            conn.execute("DROP TABLE facts")
    # Databases from before the XBRL check lack its column
    columns = {row[1] for row in conn.execute("PRAGMA table_info(figures)")}
    if "xbrl" not in columns:
        with conn:
            conn.execute("ALTER TABLE figures ADD COLUMN xbrl TEXT NOT NULL DEFAULT ''")
    return conn


def replace_facts(owner: str, source_file: str, facts: list[Fact]) -> None:
    """Replace everything stored for one owner's file with `facts`."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM figures WHERE owner = ? AND source_file = ?", (owner, source_file)
        )
        conn.executemany(
            "INSERT INTO figures (owner, source_file, metric, fiscal_year, value, unit, "
            "label, page, line, segment, xbrl) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (f.owner, f.source_file, f.metric, f.fiscal_year, str(f.value), f.unit,
                 f.label, f.page, f.line, f.segment, f.xbrl)
                for f in facts
            ],
        )
    conn.close()


def load_facts(owners: list[str], source_file: str | None = None) -> list[Fact]:
    """Facts belonging to any of `owners`, optionally for one file."""
    if not FACTS_DB.exists() or not owners:
        return []
    query = (
        "SELECT owner, source_file, metric, fiscal_year, value, unit, label, page, line, "
        f"segment, xbrl FROM figures WHERE owner IN ({','.join('?' * len(owners))})"
    )
    params: list = list(owners)
    if source_file is not None:
        query += " AND source_file = ?"
        params.append(source_file)
    with _connect() as conn:
        rows = conn.execute(
            query + " ORDER BY source_file, metric, segment, fiscal_year", params
        ).fetchall()
    conn.close()
    return [
        Fact(owner, source, metric, year, Decimal(value), unit, label, page, line, segment, xbrl)
        for owner, source, metric, year, value, unit, label, page, line, segment, xbrl in rows
    ]


def count_facts(owner: str) -> dict[str, int]:
    """Number of stored figures per source file for one owner."""
    if not FACTS_DB.exists():
        return {}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT source_file, COUNT(*) FROM figures WHERE owner = ? GROUP BY source_file",
            (owner,),
        ).fetchall()
    conn.close()
    return dict(rows)


@traced("extract_figures")
def extract_and_store(
    pages: dict[int, str], source_file: str, owner: str, company: str | None = None,
    cik: int | None = None,
) -> int:
    """Re-extract one file's figures, replacing any stored before. Returns the count.

    Old figures are cleared even if extraction fails, so a replaced filing never
    keeps the previous version's numbers. When `company` (the registrant on the
    cover page) is an SEC filer, the figures are checked against its XBRL data:
    confirmed ones are marked, contradicted ones dropped or re-read from the row
    that prints XBRL's values (see xbrl.py).
    """
    replace_facts(owner, source_file, [])
    facts = extract_facts(pages, source_file, owner)
    if facts:
        try:
            facts += extract_breakdowns(pages, source_file, owner, facts)
        except Exception as exc:
            # Breakdowns are an extra: the statement figures still stand
            print(f"Warning: breakdown extraction failed for {source_file}: {exc}")
    if facts:
        from financial_rag import xbrl  # local import: xbrl imports this module

        report = xbrl.crosscheck(facts, company, cik, statement_pages(pages))
        print(xbrl.summary(report, source_file))
        facts = report.facts
    replace_facts(owner, source_file, facts)
    note(source_file=source_file, figures=len(facts))
    return len(facts)


def load_pages(path: Path) -> dict[int, str]:
    """A filing's page text by page number: a PDF, or an EDGAR HTML 10-K."""
    if path.suffix.lower() in (".htm", ".html"):
        from financial_rag import html_filing

        html = path.read_text(encoding="utf-8", errors="replace")
        return {page.number: page.text for page in html_filing.load_pages(html)}
    from langchain_community.document_loaders import PyPDFLoader

    return {doc.metadata["page"]: doc.page_content for doc in PyPDFLoader(str(path)).load()}


def main(argv: list[str]) -> None:
    if argv:
        for path in map(Path, argv):
            pages = load_pages(path)
            facts = extract_facts(pages, path.name, "local")
            facts += extract_breakdowns(pages, path.name, "local", facts)
            for fact in facts:
                print(f"{fact.metric:28} {fact.segment[:24]:24} FY{fact.fiscal_year}  "
                      f"{fact.value:>20}  p.{page_number(fact.page)}  {fact.line}")
        return

    from financial_rag import ingest  # local import: ingest imports this module

    for pdf_path, owner in ingest.filing_jobs():
        pages = load_pages(pdf_path)
        extract_and_store(pages, pdf_path.name, owner, ingest.registrant_name(pages.get(0, "")))


if __name__ == "__main__":
    main(sys.argv[1:])
