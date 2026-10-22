"""
router.py — Send each question to the engine that can answer it properly.

Questions about a reported amount, margin, ratio, growth rate or year-over-year
comparison go to the deterministic analysis engine (analysis.py), which
computes from structured figures and cites every input. Everything else goes
to text retrieval (rag.py).

The model's only job here is to read the question into a plan: which metrics
from a fixed catalog, which fiscal years, which filing. The numbers and the
answer text are produced by code, so the model never writes a figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from financial_rag import analysis, config, financials, followup, ingest, rag, tracing
from financial_rag.ingest import PUBLIC_OWNER

# The planner picks from the company-wide metrics only; a part of the company
# (a segment, an area, a product line) comes through the qualifier, and code
# maps it onto a breakdown metric (breakdown_plan).
PLANNER_METRICS = {k: m for k, m in analysis.METRICS.items() if m.dimension is None}
MetricKey = Literal[tuple(PLANNER_METRICS)]  # type: ignore[valid-type]


class QuestionPlan(BaseModel):
    route: Literal["calculation", "text"]
    metrics: list[MetricKey]
    fiscal_years: list[int]
    year_over_year: bool
    # Copied from the question, not chosen from the filing list: matching it to a
    # filing is done in code (match_filing), which the model was unreliable at.
    companies: list[str] = Field(
        description="Each company whose own filing the question asks about, as the question "
                    "names it (or via its product or ticker): two or more for a comparison. "
                    "Not an acquisition, competitor, customer or partner it merely mentions. "
                    "Empty if it doesn't name one."
    )
    # Also copied, not decided: any qualifier sends the question to text retrieval,
    # because the engine only holds company-wide annual figures. Left to the model,
    # "Intelligent Cloud revenue" was answered with total revenue.
    qualifier: str | None = Field(
        description="Anything in the question that narrows or changes what a listed metric "
                    "measures: a segment, product, geography, quarter, adjusted or non-GAAP "
                    "measure, or a different version of the item. null if none."
    )


class UnknownFiling(Exception):
    """The caller asked about a filing they can't see (or that doesn't exist)."""


@dataclass
class Answer:
    answer: str
    sources: list[dict]
    route: str                                   # calculation | text | none
    series: list[analysis.Series] = field(default_factory=list)
    filing: str | None = None                    # the one filing searched, if narrowed
    question: str = ""                           # as answered: follow-ups made standalone
    # The filings searched when narrowed (one, or several for a comparison); empty
    # when every visible filing was
    filings: list[str] = field(default_factory=list)


PLANNER_PROMPT = """You route questions about company financial filings.

Choose route "calculation" whenever the question asks for the value of a metric listed \
below: a reported amount (such as revenue or net income), a margin, ratio or return, its \
growth, or a comparison between years. This holds in any wording or tense and whether or \
not a year is named: with no year, the latest year in the filing is used.
Choose "text" when the question asks why or how something happened, or about anything \
not in the list: risks, strategy, accounting policies, other amounts, and counts of \
anything other than shares (employees, customers, stores). A listed metric for \
one part of the company (a segment, product line, geography) is still "calculation", with \
that part in qualifier.

For "calculation":
- metrics: the keys of the metrics asked about ("operating margin" -> operating_margin; \
"revenue growth" -> revenue with year_over_year true).
- fiscal_years: only the fiscal years named in the question, as integers \
("fiscal 2024" -> 2024). Empty if none is named. For a range ("from 2022 to 2024") list \
every year in it. Never add a year just to compare with: set year_over_year instead.
- year_over_year: true if the question asks about growth, change, an increase or decrease, \
or a comparison with the prior year.

For both routes:
- companies: each company whose own filing the question asks about, when the question \
names it ("Microsoft's" -> ["Microsoft"]) or names its product or ticker ("Azure" -> \
["Microsoft"], "MSFT" -> ["Microsoft"]). A comparison names several ("Apple vs \
Microsoft" -> ["Apple", "Microsoft"]). Not a company that is only mentioned as an \
acquisition, competitor, customer, supplier or partner of the company being asked about. \
Empty when the question doesn't name the company it asks about.
- qualifier: anything in the question that narrows or changes what a listed metric \
measures, copied from the question: a segment or product ("Intelligent Cloud", \
"LinkedIn"), a geography ("United States"), a quarter ("Q4"), an adjusted or non-GAAP \
measure ("adjusted operating income"), or a different version of the item ("dividends \
declared", when the list has dividends paid). null when the question asks for a listed \
company-wide annual metric as it is, in its own words or the list's. A company name or a \
fiscal year is not a qualifier.

For "text": metrics and fiscal_years empty, year_over_year false.

Examples:
- "What is the current ratio?" -> calculation, [current_ratio], [], false
- "Return on equity?" -> calculation, [return_on_equity], [], false
- "What was the operating margin in 2024 and how did it change?" -> calculation, \
[operating_margin], [2024], true
- "How did revenue grow from 2022 to 2024?" -> calculation, [revenue], [2022, 2023, 2024], true
- "Gross and net margin for fiscal 2023" -> calculation, [gross_margin, net_margin], [2023], false
- "Why did operating margin improve?" -> text
- "How many employees did Acme have?" -> text, companies ["Acme"] (a headcount, not shares)
- "How fast did the cloud segment grow?" -> calculation, [revenue], [], true, qualifier \
"cloud segment"
- "What risks does Acme describe?" -> text, companies ["Acme"]
- "What was Microsoft's net margin?" -> calculation, [net_margin], companies ["Microsoft"]
- "What was revenue?" -> calculation, [revenue], companies []
- "How much goodwill came from the Activision Blizzard acquisition?" -> text, companies [] \
(Activision Blizzard is what was acquired, not the company asked about)
- "How does Microsoft compete with Google?" -> text, companies ["Microsoft"] (Google is a \
competitor it mentions, not a company whose filing is asked about)
- "Compare Apple's and Microsoft's operating margins" -> calculation, [operating_margin], \
companies ["Apple", "Microsoft"]
- "Which had more revenue, Acme or Microsoft?" -> calculation, [revenue], companies \
["Acme", "Microsoft"]
- "How do Apple's and Microsoft's AI risks differ?" -> text, companies ["Apple", "Microsoft"]
- "What was Intelligent Cloud revenue in 2024?" -> qualifier "Intelligent Cloud"
- "What were dividends declared in fiscal 2024?" -> qualifier "declared"
- "What were dividends paid in fiscal 2024?" -> qualifier null (that is the listed metric)
- "How much was left in the share repurchase program?" -> text (what remains of an \
authorization is not share_repurchases, the cash spent on repurchases in a year)
- "What was the operating margin in fiscal 2024?" -> qualifier null

Metrics:
{metrics}
"""


def visible_filings(owner: str) -> dict[str, ingest.Filing]:
    """The caller's filings plus the samples, by file name. As with figures, a
    user's own file shadows a sample of the same name."""
    out: dict[str, ingest.Filing] = {}
    for f in ingest.filings([owner, PUBLIC_OWNER]):
        if f.source_file not in out or f.owner == owner:
            out[f.source_file] = f
    return out


# Words that don't tell one company's filing from another's
_GENERIC_NAME_WORDS = frozenset(
    "inc incorporated corp corporation co company companies holdings holding group ltd "
    "limited plc llc lp sa ag nv the and of sec form annual report filing pdf 10k".split()
)


def _name_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in _GENERIC_NAME_WORDS}


def filings_named(company: str | None, filings: dict[str, ingest.Filing]) -> list[str]:
    """
    The filings whose company or file name shares a distinctive word with the
    company the question names ("Microsoft's" matches "Microsoft Corporation" and
    microsoft-sec-10k.pdf). Empty when the question names no company, or one
    with no filing here.
    """
    wanted = _name_words(company or "")
    if not wanted:
        return []
    return [
        name for name, filing in filings.items()
        if wanted & (_name_words(filing.company or "") | _name_words(name.rsplit(".", 1)[0]))
    ]


def company_of(filings: dict[str, ingest.Filing]) -> dict[str, str]:
    """
    Each filing's company, by its distinctive name words, so two 10-Ks of one
    company ("Microsoft Corporation", "MICROSOFT CORP") merge into one history.
    A filing whose company is unknown stands alone.
    """
    out = {}
    for name, filing in filings.items():
        words = _name_words(filing.company or "")
        if words:
            out[name] = " ".join(sorted(words))
    return out


def match_filing(company: str | None, filings: dict[str, ingest.Filing]) -> str | None:
    """The one filing the named company matches; None for no match or several."""
    hits = filings_named(company, filings)
    return hits[0] if len(hits) == 1 else None


# Words that don't single out a part: "the cloud segment" means "cloud"
_GENERIC_PART_WORDS = frozenset(
    "segment segments business businesses division unit line lines the and of our its "
    "revenue revenues sales operating income margin".split()
)
# A qualifier naming the kind of part restricts the match to that kind
_DIMENSION_WORDS = {
    "segment": {"segment", "segments"},
    "geography": {"country", "countries", "geographic", "geography", "region", "regions"},
    "product": {"product", "products", "offering", "offerings"},
}
MAX_PARTS = 2   # a qualifier matching more parts than this is too vague to calculate

# Abbreviated country names, as filings print them ("U.S.", "UK"); case-sensitive,
# so the word "us" is left alone
_COUNTRY_ABBREVIATIONS = [
    (re.compile(r"\bU\.S\.(?:A\.)?|\bUSA?\b"), "United States"),
    (re.compile(r"\bU\.K\.|\bUK\b"), "United Kingdom"),
]


def _part_words(text: str) -> set[str]:
    """Words of a part's name or a qualifier: "U.S." and "United States" give the same."""
    for pattern, name in _COUNTRY_ABBREVIATIONS:
        text = pattern.sub(name, text)
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def breakdown_plan(
    metrics: list[str], qualifier: str, facts: list[financials.Fact]
) -> tuple[list[str], list[str]] | None:
    """
    Map company-wide metrics and a qualifier onto breakdown metrics and parts:
    (["revenue"], "Intelligent Cloud") -> (["segment_revenue"], ["Intelligent Cloud"]).

    A part matches when every distinctive word of the qualifier is in its name
    ("Dynamics" matches "Dynamics products and cloud services"). None when a
    metric has no breakdown, nothing matches, or the match is too vague.
    """
    words = _part_words(qualifier)
    wanted = words - _GENERIC_PART_WORDS - {w for ws in _DIMENSION_WORDS.values() for w in ws}
    kinds = {d for d, ws in _DIMENSION_WORDS.items() if words & ws}
    if not wanted:
        return None
    keys: list[str] = []
    parts: set[str] = set()
    for metric in metrics:
        found = False
        for key in analysis.BREAKDOWN_OF.get(metric, []):
            breakdown = analysis.METRICS[key]
            if kinds and breakdown.dimension not in kinds:
                continue
            source = breakdown.formulas[0].inputs[0] if breakdown.formulas else key
            matched = {
                f.segment for f in facts
                if f.metric == source and wanted <= _part_words(f.segment)
            }
            if matched:
                found = True
                keys.append(key)
                parts |= matched
        if not found:
            return None
    if not parts or len(parts) > MAX_PARTS:
        return None
    return keys, sorted(parts)


# Metrics the planner reaches for when a question merely counts something. "How
# many employees did Apple have?" came back as diluted_shares 8 times in 8, so a
# share count is used only when the question says "share" (or EPS).
_METRIC_NEEDS_WORD = {
    "diluted_shares": re.compile(r"\bshares?\b|\beps\b", re.IGNORECASE),
}


def plausible_metrics(question: str, metrics: list[str]) -> list[str]:
    """The planned metrics, deduplicated, minus any the question's wording rules out."""
    return [
        m for m in dict.fromkeys(metrics)
        if m not in _METRIC_NEEDS_WORD or _METRIC_NEEDS_WORD[m].search(question)
    ]


def _joined(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def _no_filing_for(companies: list[str], filings: dict[str, ingest.Filing]) -> Answer:
    available = "; ".join(
        f"{f.company} ({name})" if f.company else name for name, f in filings.items()
    )
    which = _joined(companies)
    return Answer(
        f"No filing for {which} is available, so I can't answer that from a filing. "
        f"Filings you can ask about: {available or 'none yet'}. Upload {which}'s 10-K "
        "to ask about it.",
        [], "none",
    )


def _missing_note(missing: list[str]) -> str:
    """Said first when a comparison names a company with no filing here."""
    if not missing:
        return ""
    return (f"_No filing for {_joined(missing)} is available, so this covers only the "
            "other companies named._\n\n")


def plan_question(question: str) -> QuestionPlan:
    """
    Read a question into a plan. The model isn't shown the user's filings: given
    a list, it tended to fill in a company the question never named.
    """
    metrics = "\n".join(
        f"- {m.key}: {m.name}. {m.description}" for m in PLANNER_METRICS.values()
    )
    llm = config.chat_model(temperature=0).with_structured_output(QuestionPlan)
    return llm.invoke([
        ("system", PLANNER_PROMPT.format(metrics=metrics)),
        ("human", question),
    ])


def _text_answer(
    question: str, owner: str, chosen: list[ingest.Filing] | None = None, preface: str = ""
) -> Answer:
    """Retrieval over the chosen filings (each given its share), or over all of them."""
    chosen = chosen or []
    if len(chosen) > 1:
        result = rag.query(question, owner,
                           filings=[(f.owner, f.source_file) for f in chosen])
    else:
        scope = (chosen[0].owner, chosen[0].source_file) if chosen else None
        result = rag.query(question, owner, scope)
    return Answer(preface + result.answer, result.sources, "text", **_narrowed(chosen))


def _narrowed(chosen: list[ingest.Filing]) -> dict:
    names = [f.source_file for f in chosen]
    return {"filing": names[0] if len(names) == 1 else None, "filings": names}


def visible_facts(owner: str) -> list[financials.Fact]:
    """The caller's figures plus the samples'. A user's own file shadows a sample
    of the same name, so the two are never mixed into one set of figures."""
    facts = financials.load_facts([owner, PUBLIC_OWNER])
    own_files = {f.source_file for f in facts if f.owner == owner}
    return [f for f in facts if f.owner == owner or f.source_file not in own_files]


def answer(
    question: str,
    owner: str,
    source_file: str | None = None,
    planner=plan_question,
    history: list[tuple[str, str]] | None = None,
    rewriter=followup.standalone_question,
) -> Answer:
    """
    Answer from the caller's filings and the samples.

    `history` (earlier (role, text) turns, oldest first) lets a follow-up such as
    "and in 2023?" be rewritten into a standalone question first; the answer
    records the question it actually answered.

    `source_file` (picked in the UI) narrows both routes to that one filing.
    Otherwise the planner narrows them to the filings of the companies the
    question names: one, or several for a comparison. A question that names none
    searches every filing, and the prompt makes the model say which filing each
    figure comes from.
    """
    with tracing.tracer.start_as_current_span("answer"):
        tracing.note(follow_up=bool(history), filing_picked=source_file is not None)
        if history:
            with tracing.tracer.start_as_current_span("rewrite_follow_up"):
                asked = rewriter(question, history)
        else:
            asked = question
        result = _answer(asked, owner, source_file, planner)
        result.question = asked
        tracing.note(route=result.route, filings=len(result.filings), series=len(result.series))
        return result


def _answer(question: str, owner: str, source_file: str | None, planner) -> Answer:
    filings = visible_filings(owner)
    if source_file is not None and source_file not in filings:
        raise UnknownFiling(f"No filing named {source_file}.")
    facts = visible_facts(owner)

    plan = None
    if facts or (source_file is None and len(filings) > 1):
        with tracing.tracer.start_as_current_span("plan"):
            try:
                plan = planner(question)
            except config.GatewayLimit:
                raise   # a spent budget is the user's answer, not a reason to degrade it
            except Exception as exc:
                print(f"Planner failed, answering from text: {exc}")
                tracing.note(failed=True)
            else:
                tracing.note(route=plan.route, metrics=plan.metrics, fiscal_years=plan.fiscal_years,
                     companies=len(plan.companies), qualified=plan.qualifier is not None)

    chosen: list[str] = [source_file] if source_file else []
    missing: list[str] = []
    if not chosen and plan and plan.companies:
        for company in dict.fromkeys(plan.companies):
            named = filings_named(company, filings)
            if not named:
                missing.append(company)
            chosen += [name for name in named if name not in chosen]
        if not chosen:
            # Answering with another company's figures would be wrong, not partial
            return _no_filing_for(missing, filings)
    narrowed = [filings[name] for name in chosen]
    note = _missing_note(missing)
    if narrowed:
        facts = [f for f in facts if f.source_file in chosen]
    metrics = plausible_metrics(question, plan.metrics) if plan else []
    if plan is None or plan.route != "calculation" or not metrics or not facts:
        return _text_answer(question, owner, narrowed, note)

    parts = None
    if plan.qualifier:
        # A part of the company the filing breaks out (a segment, an area, a
        # product line) is calculated; anything else the qualifier names (a
        # quarter, an adjusted measure, "declared" vs paid) isn't in the
        # engine's figures, so it goes to text.
        breakdown = breakdown_plan(metrics, plan.qualifier, facts)
        if breakdown is None:
            return _text_answer(question, owner, narrowed, note)
        metrics, parts = breakdown
    with tracing.tracer.start_as_current_span("calculate"):
        series = analysis.analyze(facts, metrics, plan.fiscal_years, plan.year_over_year,
                                  parts, company_of(filings))
        tracing.note(metrics=metrics, series=len(series),
             results=sum(r.value is not None for s in series for r in s.results))

    if not any(r.value is not None for s in series for r in s.results):
        # Nothing computable: say why, then answer from the text without arithmetic.
        problems = sorted({
            f"{s.metric.name} FY{r.fiscal_year}: {r.problem}"
            for s in series for r in s.results if r.problem
        })
        preface = (
            note
            + "I couldn't calculate this from the structured figures"
            + (":\n" + "\n".join(f"- {p}" for p in problems) if problems else ".")
            + "\n\nFrom the filing text instead:\n\n"
        )
        return _text_answer(question, owner, narrowed, preface)

    names = {name: f.company for name, f in filings.items() if f.company}
    markdown, sources = analysis.render(series, names)
    return Answer(note + markdown, sources, "calculation", series, **_narrowed(narrowed))
