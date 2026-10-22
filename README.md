# 📊 Financial Report Q&A Engine

A question-answering system for SEC 10-K filings. Questions about figures (margins,
ratios, growth rates, year-over-year changes) are answered by a **deterministic analysis
engine** that computes in Python from figures read out of the financial statements, citing
the page and row of every input. Everything else goes through retrieval-augmented
generation (RAG): it retrieves the passages that actually answer your question and returns
an answer with page-level citations, refusing to guess when the figure isn't in the
documents.

Built with LangChain, ChromaDB, OpenAI (`gpt-4o-mini` + `text-embedding-3-small`),
hybrid semantic + BM25 keyword search, two-stage reranking (cross-encoder, then
gpt-4o-mini), FastAPI and Streamlit.

---

## Why it doesn't hallucinate numbers

Financial Q&A fails in a specific way: the model quietly derives a figure that never
appears in the filing. Five parts of the pipeline are aimed at that.

**Calculations never touch the model.** "What was the operating margin, and how did it
change?" is arithmetic over specific line items, not a retrieval problem. At ingest, the
financial statements are extracted into structured records; questions about a metric are
computed with `Decimal` arithmetic and written out by code, so a language model never
produces a figure. See [Financial analysis engine](#financial-analysis-engine).

**Table-aware chunking.** 10-K financials are tables that a naive splitter cuts in half,
stranding a number from its row label. `ingest.py` detects contiguous table-like lines
(dollar amounts, comma-grouped numbers, dash separators) and hides their newlines behind
a word-joiner placeholder so the token splitter treats each table block as one unit,
then restores them after splitting.

**Hybrid retrieval + reranking.** Embeddings find passages that mean the same thing as the
question but blur exact tokens: fiscal years, line-item labels, precise values. So two
searches run side by side: semantic (20 candidates above a relevance floor) and BM25
keyword search (20 more) that treats "245,122", "$245,122" and "245122" as one token and
"FY24" as "fiscal 2024". Their rankings are fused, a cross-encoder and then gpt-4o-mini
rerank the union, and only the top 5 go to the answering model. Every citation records
how its chunk was found. If neither search finds anything, the API says so instead of answering from
parametric memory. See [Hybrid retrieval](#hybrid-retrieval).

**A prompt that forbids arithmetic.** For questions that go to text retrieval, the system
prompt allows only figures that appear verbatim in the retrieved excerpts — no computing, deriving, or combining rows — and
requires a `(file.pdf, p.NN)` citation after each claim. Anything missing comes back as
"not found in context".

**…and code that checks the answer anyway.** The prompt isn't enough on its own. Asked
for Activision Blizzard's goodwill, gpt-4o-mini was given "Goodwill 50,969" (p.90) and
"$51.0 billion" (p.94), and answered "$51,969 million (p.92)" five times out of five: a
blend of the two, cited to a page that holds neither. So `rag.check_answer` compares every
text answer against its excerpts before it is returned:
- every comma-grouped or decimal figure must be printed on the page cited right after it;
- every citation must name a retrieved page.

If the check fails, the model is asked once more with the problems spelled out. That
fixed the goodwill answer to "$51.0 billion (p.94)". If the retry fails too, the answer
carries a visible "Not verified against the excerpts" warning naming each problem.

---

## Architecture

```
                  ingest.py
   data/*.pdf ──► load ─┬─► table-aware split ─► embed ──────────► chroma_db/
                        └─► statement pages ─► map rows ─► verify ─► figures (SQLite)
                                                                         │
   question ──► router.py: plan (metric? years? filing?)                 │
                   │                                                     │
                   ├─ metric ──► analysis.py: Decimal math over figures ◄┘
                   │                 └─► answer with every input cited (page + row)
                   │
                   └─ other ───► semantic search (k=20) + BM25 keyword search (k=20)
                                    ─► rank fusion ─► cross-encoder ─► LLM rerank (top 5)
                                    ─► gpt-4o-mini + strict prompt ─► answer + citations

                          api.py (FastAPI :8000) ◄── app.py (Streamlit :8501)
```

The modules live in the `financial_rag/` package. Run them from the repository root
(`python -m financial_rag.<module>`), since data and the vector store are found by
relative paths.

| File | Role |
| --- | --- |
| `ingest.py` | Loads filings (PDF, or EDGAR's HTML 10-K) from `data/`, chunks them (1024 tokens / 128 overlap, table-aware), embeds and persists to ChromaDB. |
| `html_filing.py` | Reads EDGAR's HTML 10-K: pages at the filing's page breaks, one line per table row, and each table's column headings. |
| `jobs.py` | Background jobs (fetching and indexing a filing by ticker): one worker thread, state kept in SQLite, `GET /jobs/{id}` for progress. |
| `financials.py` | Extracts the income statement, balance sheet and cash flow statement into verified, cited figures; stores them in SQLite. |
| `analysis.py` | Deterministic calculations (margins, ratios, returns, growth, YoY, CAGR) over those figures, and the cited write-up. |
| `router.py` | Plans each question, narrows it to the filing it names, and sends it to the analysis engine or to text retrieval. |
| `followup.py` | Rewrites a follow-up ("And in 2023?") into a standalone question using the last few turns. |
| `rag.py` | Hybrid retrieval, score-threshold filtering, cross-encoder reranking, prompting. Exposes `query()`. |
| `hybrid.py` | The keyword half: financial-aware tokenizer, BM25, query profile, rank fusion. Pure functions. |
| `eval_retrieval.py` | Retrieval benchmark: three pipeline configurations on 72 questions, scored on whether the right passage is retrieved. |
| `eval_answers.py` | End-to-end benchmark: 52 questions on two companies' filings through the whole pipeline, scored on whether the final answer is right, routed, verified and cited. |
| `api.py` | FastAPI service — `POST /query`, `POST /documents` (upload), `POST /filings` (fetch by ticker and year), `GET /jobs/{id}`, `GET /documents`, `GET /financials`, `GET /usage`, `GET /health`. |
| `usage.py` | Usage limits: questions and uploads per user per month, questions per minute, and a daily cap across all users. |
| `edgar.py` | SEC EDGAR client: ticker and company lookup, filings, XBRL company facts; identified, rate-limited and cached. |
| `xbrl.py` | Checks each extracted figure against the company's XBRL data: matches are marked, contradictions dropped. |
| `eval_xbrl.py` | Benchmark generated from XBRL for any tickers: fetches each latest 10-K, asks for reported and derived figures, scores against the SEC's values. |
| `app.py` | Streamlit chat UI with expandable source citations and a sidebar for uploading PDFs. |
| `evaluate.py` | Optional RAGAs evaluation (faithfulness, answer relevancy) with a before/after comparison table. |
| `config.py` | The models used: `CHAT_MODEL` for every language-model call, the embedding model and the reranker. |
| `start.py` | One-command boot: checks config, ingests if needed, starts both servers, opens the browser. |
| `gateway/` | Optional LLM gateway in Go between the API and the providers: limits, budgets, cache, fallback, redaction, metrics (see [LLM gateway](#llm-gateway)). |

---

## Financial analysis engine

Ask "What was the free cash flow margin in 2024 and how did it change from 2023?" and the
answer is computed, not generated:

```
Free cash flow margin · microsoft-sec-10k.htm
- FY2023: 28.1% = (Net cash from operating activities − Capital expenditures) ÷ Total revenue, from:
    - Net cash from operating activities ("Net cash from operations") FY2023: $87,582 million (microsoft-sec-10k.htm, p.59; matches SEC XBRL data)
    - Capital expenditures ("Additions to property and equipment") FY2023: $28,107 million (microsoft-sec-10k.htm, p.59; matches SEC XBRL data)
    - Total revenue FY2023: $211,915 million (microsoft-sec-10k.htm, p.56; matches SEC XBRL data)
- FY2024: 30.2% = … (same inputs for FY2024)
- Change FY2023 → FY2024: +2.2 percentage points
```

**1. Extraction (`financials.py`, once per filing at ingest).** Code finds the three
primary statements: pages whose heading ("INCOME STATEMENTS", "BALANCE SHEETS", "CASH FLOWS
STATEMENTS" and the usual variants) sits near the top and that are dense with figures.
`gpt-4o-mini` then maps each of 22 canonical line items (revenue, operating income, total
assets, capital expenditures, …) to the row that reports it: page, label as printed, and
one sample value with its fiscal year. **The model's values are only used to check its
mapping, never stored.** Code then verifies the mapping against the page text:

- the row must exist on that page and its label must match (wrapped labels included);
- the sample value must sit in the column whose header year matches (`Year Ended June 30,
  2024 2023 2022`, or Apple's dated columns, which PDF extraction puts one year per line),
  which catches a value attributed to the wrong year;
- if two equally good rows disagree, or two mappings for one metric give different values,
  the metric is dropped rather than guessed.

Only then are the values read, for every year column, from the PDF text itself. They are
scaled by the statement's own "(In millions, except per share amounts)" note (amounts are
rejected if the page states no scale), and outflows printed in parentheses are stored as
positive amounts. Share counts follow the note's own wording for shares: Apple's "(In
millions, except number of shares, which are reflected in thousands…)" puts them in
thousands, and "(In thousands, except share data)" leaves them unscaled. Every figure
keeps its file, page and row as printed. On Microsoft's FY2024 10-K this yields 56-59
figures (21-22 line items, two or three years each), and on Apple's FY2025 10-K 59, with
every mapping verified. Which items the model maps can vary a little between runs. For
example, "Diluted 7,469", which wraps under its heading, is sometimes skipped. A skipped
item is reported as missing, never guessed.

**Checked against the SEC's own data.** Every 10-K also files its statements as XBRL:
machine-readable values the company tags itself, which the SEC publishes per company.
When the cover page names an SEC registrant, `xbrl.py` looks up each extracted figure in
that data (EDGAR's companyfacts API). It uses the line item's us-gaap tags, the fiscal year
of the period's end date, and a full-year period for income and cash-flow items.
- A figure XBRL reports the same is marked, and answers cite it as "matches SEC XBRL data".
- A figure XBRL reports differently is dropped, never stored.
- A figure XBRL has no value for keeps its page check alone.

On the two sample filings every statement figure matches: Apple 59 of 59 and Microsoft
56 of 56. That includes Apple's diluted share count, which XBRL confirms as 15,004,697,000.
Read at the dollars' scale of millions, it would have been 1,000× too large and dropped.
Segment, country and product breakdowns are dimensional XBRL, which companyfacts doesn't
include, so they stand on their add-up check.

EDGAR requires every automated request to identify who is asking. Set
`SEC_USER_AGENT="<app name> <your email>"` in `.env`. Without it, nothing is fetched and
figures are stored on the page check alone. Requests stay under the SEC's 10 a second,
and responses are cached in `chroma_db/edgar_cache/`.

**Breakdowns.** Totals are also broken down from the notes, extracted the same way:
- revenue and operating income by reportable segment;
- revenue by geography;
- revenue by product or service line.

The model points at each part's row. Code verifies the row, then keeps a breakdown only if
its parts add up exactly to the verified company total for every year. A wrong table
(long-lived assets by country, say), a missing part, or a row taken from the wrong block
breaks the sum, and the whole breakdown is dropped. Rows that PDF extraction splits across
lines ("Gaming 21,503", then "15,466", then "16,230") are rejoined first. Microsoft's 10-K
yields 54 breakdown figures:
- 3 segments for revenue and operating income;
- the United States and other countries;
- 10 product lines;

all three fiscal years each. Apple's yields 24: the United States, China and other
countries, and 5 product lines (iPhone, Mac, iPad, Wearables, Services). Apple's segments
are not extracted. Its segment tables put the segments in columns, one table per year,
and the engine reads breakdowns row by row, so segment questions go to text.

**2. Calculation (`analysis.py`).** Pure Python with `Decimal` arithmetic: no model, no
floating-point rounding until display. Results carry their inputs, and inputs carry their
citations.

| Kind | Metrics |
| --- | --- |
| Reported line items | revenue, cost of revenue, gross profit, R&D, operating income, pre-tax income, income tax, net income, basic/diluted EPS, diluted shares, cash, current assets, total assets, current liabilities, total liabilities, long-term debt, stockholders' equity, operating cash flow, capital expenditures, dividends paid, share repurchases |
| Margins | gross margin (falls back to (revenue − cost of revenue) ÷ revenue), operating margin, net margin, R&D intensity, free cash flow margin, effective tax rate |
| Ratios and returns | current ratio, long-term debt to equity, total liabilities to equity, return on equity, return on assets, dividend payout ratio |
| Derived amounts | free cash flow |
| Parts of the company | segment revenue, segment operating income, segment operating margin, revenue by geography, revenue by product or service line |
| Over time | year-over-year change for any metric (percentage change for amounts, percentage points for margins), compound annual growth over multi-year spans |

If an input is missing for a year, the answer says which one instead of producing a number.
Return on equity and return on assets use year-end balances; their formulas say so.

**3. Routing (`router.py`).** A model call reads the question into a plan: which metrics
(constrained to the catalog above), which fiscal years, which filing. If the plan names a
metric, the engine computes and code writes the answer from a template. Otherwise, or if
nothing in the plan can be computed (with the reason stated first), the question goes to
text retrieval. "Why did operating margin improve?" goes to retrieval: it asks for an
explanation, not a figure.

The planner also copies any **qualifier** out of the question: a segment or product
("Intelligent Cloud revenue"), a geography, a quarter, an adjusted measure, or a different
version of an item ("dividends *declared*", when the engine has dividends *paid*). Code,
not the model, decides what happens next (`router.breakdown_plan`):
- A qualifier naming a part the filing breaks out is calculated from that breakdown. A part
  matches when every distinctive word of the qualifier is in its name, so "Intelligent
  Cloud", "the cloud segment", "United States" and "Dynamics" all work.
- Anything else goes to text retrieval: "Azure", "Microsoft Cloud", "Q4", "adjusted",
  "declared", or a qualifier that matches more than two parts. The engine has no figure
  for any of those.

Before qualifiers existed, the end-to-end evaluation caught "Intelligent Cloud revenue",
"LinkedIn revenue" and "dividends declared" being answered with total revenue and
dividends paid, each under the "nothing is estimated" footer.

Routing is the one model-dependent step, so it has its own opt-in test
(`tests/test_planner_live.py`). It passes 47 of 47 cases, three runs in a row:
- the 20 routing questions;
- 11 on which filing a question is about;
- 12 on qualifiers;
- 4 on planning segment questions.

Figures are stored in `chroma_db/financials.sqlite3`, scoped by owner exactly like the
text chunks: users see their own filings' figures and the shared samples', never another
user's. To extract figures for filings ingested before this engine existed, without
re-embedding them:

```bash
python -m financial_rag.financials                    # (re)extract figures for every PDF in data/
python -m financial_rag.financials data/some-10k.pdf  # print what would be extracted from one PDF
```

---

## EDGAR HTML filings

A 10-K is filed with the SEC as HTML with inline XBRL, and that version keeps what a PDF's
text loses: which cells form a row, and which heading sits over which column.
`html_filing.py` reads it into the same shape as a PDF's pages:
- **Pages** split where the filing breaks pages, so citations keep page numbers. These are
  the filing's own pages, not a PDF's.
- **Tables:** one line per table row, so the figure extractor reads HTML statements with no
  changes. Hidden content (the XBRL header) is skipped.
- **Column headings:** each table's headings are recorded. A chunk that holds a table's rows
  but not its headings gets them repeated at the top, as `[Table columns: June 30, 2024
  2023]`. A PDF chunk that starts mid-balance-sheet can't say which column is 2024; this
  one can.

**Fetch by ticker.** In the sidebar, enter a ticker under "Or fetch a company's 10-K", and
optionally a fiscal year (or `POST /filings {"ticker": "AAPL", "fiscal_year": 2024}`). The
API looks the ticker up on EDGAR straight away:
an unknown ticker gets `404` and nothing is counted. Then it queues a job and answers `202`
with the job's id. The job:
1. finds the company's latest 10-K, or the one for that fiscal year as the company names
   it (Target's fiscal 2024 is the year to February 1, 2025);
2. downloads it;
3. indexes it like an upload, into your own filings, as `aapl-10k-2025-09-27.htm`;
4. checks its figures against the company's XBRL data, using the CIK from the ticker.

The UI follows `GET /jobs/{id}` and shows each step until it reports the filing's chunks and
figures. A fetch counts against the monthly upload limit and is refunded if it fails.
Fetch a company's earlier years too, and its filings merge into one history (see
[Which filing a question is about](#which-filing-a-question-is-about)).

Jobs run one at a time on a worker thread (`jobs.py`), and their records are kept in
`chroma_db/jobs.sqlite3`. A job interrupted by a server restart is marked failed, never left
"running". You can also drop any `.htm`/`.html` filing in `data/`, and `python -m
financial_rag.ingest` picks it up.

**Results.** The samples are now these HTML 10-Ks rather than PDFs of the same filings:
- **Microsoft:** 59 statement figures, all 59 confirmed by XBRL; the PDF gave 56.
- **Apple:** 59 of 59 confirmed. Its country breakdown, which the PDF runs sometimes
  missed, extracts, so "How much of Apple's revenue came from the United States?" is
  calculated rather than read from text.
- **Benchmark:** 47/47 correct, routed, verified and cited. In one of two runs the
  Activision goodwill question was answered from the goodwill table's "Acquisitions"
  column ($51,235 million, every FY2024 acquisition) rather than Note 8's $50,969 million.
  "Short-term unearned revenue", the question the PDF version missed for want of its
  column headings, is answered.
- Citations point at the HTML filing's own page breaks, so page numbers differ from the
  PDFs'.

**A skipped part is completed, not guessed.** The model sometimes leaves out a small row
when mapping a breakdown: Microsoft's "Other" product line, $45 million of $245 billion.
The add-up check then rejects the whole breakdown. Now, when the parts fall short, the one
row on the same page that reports the shortfall in every year is added. A gap that no
single row fills, or that two rows could fill, is never filled. An overshoot (a wrong row
mapped) leaves a negative gap, so a wrong mapping is still rejected.

## Hybrid retrieval

Financial questions are full of exact tokens: "fiscal 2024", "short-term unearned
revenue", "57,582". Embeddings capture meaning but treat those loosely. In the benchmark
below, semantic search ranked the balance-sheet chunk with short-term unearned revenue
only 11th, and never found the passage with the employee count. So `rag.py` runs two
searches over the same chunks and combines them.

1. **Semantic.** The question is embedded, and the top 20 chunks above `SCORE_THRESHOLD`
   are kept, as before.
2. **Keyword (BM25).** `hybrid.py` scores every chunk the caller may see with Okapi BM25.
   Its tokenizer is built for filings:
   - numbers lose their grouping commas and `$`, so "$ 245,122", "245,122" and "245122"
     all match;
   - "FY24", "FY2024" and "FY '24" become "fiscal 2024", the way filings write it;
   - plurals fold ("liabilities" matches "liability") and stopwords drop.

   The index is built lazily from ChromaDB and cached in the API process. Each query reads
   only chunks added since the last one, and replaced filings drop out.
3. **Query-aware fusion.** The two rankings are merged with weighted reciprocal rank
   fusion. The keyword ranking weighs more when the question quotes a value (2.0), the
   same as semantic when it names a year (1.0), and less for purely conceptual questions
   (0.75).
4. **Rerank, in two stages.** The cross-encoder scores the top 30 fused candidates.
   Then gpt-4o-mini orders the cross-encoder's top 15: it reads statements, line items
   and fiscal years far better than the small cross-encoder, which was trained on web
   search. The rest keep the cross-encoder's order after the LLM's picks. If the LLM call
   fails or times out, the cross-encoder's order is used. Either way, when the question
   quotes exact values, chunks containing all of them go first. The top 5 go to the
   answering model. This step costs about 1.3 seconds and $0.002 per question; see
   [Retrieval benchmark](#retrieval-benchmark) for why it's worth it.

**Provenance.** Every retrieved chunk carries a `retrieval` record, and the API returns it
with each citation. It says:
- which search found the chunk (`matched_by`), with each search's rank and score;
- the query terms it matched;
- the cross-encoder score, which reranker set the final order (`reranker`), and the
  LLM's rank for the chunk (`llm_rank`) when it did;
- whether it contains the quoted values;
- Chroma's `chunk_id`.

The UI shows a one-line summary under each citation. The citation excerpt is the
stretch of the chunk that matches the question, so the figure is visible.

**Isolation.** The keyword path is scoped by the same owner filter as the semantic one,
applied by Chroma before any scoring. Term statistics come only from chunks the caller
may see, so one user's filings never influence another's results. Tests cover both.

`HYBRID_SEARCH=0` switches back to the semantic-only pipeline, unchanged from before.
`RERANKER=cross-encoder` drops the LLM reranking step.

---

## Follow-up questions

The chat keeps context. "What was Microsoft's net income in fiscal 2024?" followed by
"And in 2023?" is read as "What was Microsoft's net income in fiscal 2023?". Then
"What about Acme?" moves the same question to Acme's filing.

- The UI sends the last six messages with each question (`history` in `POST /query`).
- `followup.py` has gpt-4o-mini rewrite the new question so it stands on its own. It
  resolves only what the question refers back to. A question that already stands alone,
  or names a new company or metric, is left as asked or replaces the earlier subject.
- The rewritten question then goes through the unchanged pipeline: planner, filing choice,
  calculation or retrieval, and the answer check.
- The UI shows "Read as: …" above the answer whenever the question was rewritten, and the
  API returns it as `question`.
- If the rewrite call fails, the question is answered as asked. Questions with no
  history make no extra call.

The opt-in live test (`RUN_LIVE_TESTS=1 pytest tests/test_followup_live.py`) passed 5 of
5 cases, three runs in a row: another year, another metric, another company, a pronoun,
and an already-standalone question.

---

## Which filing a question is about

Every user sees the shared Microsoft sample alongside their own uploads. Without care,
"What was revenue in 2024?" from someone who uploaded another company's 10-K would mix
two companies' passages, and the model would have to guess which figure to give.

- **Company names.** Each filing's company is read from its cover page: the line above
  "(Exact name of registrant as specified in its charter)", or after "Commission File
  Number" on covers like Microsoft's. The sidebar and `GET /documents` show it. Filings
  ingested earlier get it the first time they're listed, with no re-ingest.
- **Naming a company narrows the search.** The planner copies the company a question
  names ("Acme's risks" → "Acme"; a product such as "Azure" gives "Microsoft"). Code then
  matches that name to a filing by the distinctive words in its company and file names.
  Both routes then use only that company's filings: the analysis engine's figures, and
  both halves of hybrid retrieval.
- **Comparisons name several.** The planner returns every company a question compares
  ("Apple vs Microsoft operating margin" → ["Apple", "Microsoft"]; "How does Microsoft
  compete with Google?" → only Microsoft, since Google is just mentioned). Each company's
  figures are calculated from its own filing, and a comparison line leads the answer,
  highest first:
  `Compared · Operating margin: Microsoft Corporation 44.6% (FY2024) · Apple Inc. 32.0%
  (FY2025) (each company's latest fiscal year)`. The latest year all of them have is
  compared, or each one's latest when they share none, since fiscal years end on
  different dates. A text comparison searches each filing separately (3 chunks each for
  two), so one company can't crowd the other out of the context. A named company with no
  filing is said to be missing and the others are answered; if none has a filing, the
  question is refused.
  - The planner isn't shown the filing list. Given one, gpt-4o-mini filled in a company
    for questions that named none, such as "What was net income in fiscal 2024?". With
    only extraction asked of it, the opt-in live test
    (`RUN_LIVE_TESTS=1 pytest tests/test_planner_live.py`) passed 28 of 28 cases, three
    runs in a row. Those cases are the 20 routing questions plus 8 on which filing to use.
- **Or pick one.** The sidebar's **Ask about** picker (`source_file` in `POST /query`)
  limits every answer to one filing, whatever the question says. A user's own file shadows
  a sample of the same name, and a filing the caller can't see is refused with `404`.
- **One company's filings make one history.** Filings whose cover pages name the same
  company ("Microsoft Corporation", "MICROSOFT CORP") are merged: two 10-Ks give four or
  five years of figures, so "Microsoft's revenue from 2022 to 2026" and its growth are
  calculated across them. Where filings overlap, the newer one (the later fiscal year)
  wins, so a restatement replaces what was first reported, and the answer says so:
  `Restated: Cost of revenue FY2022 is $82,306 million in tgt-10k-2025-02-01.htm, the
  newer filing; tgt-10k-2024-02-03.htm reported $82,229 million.` Every figure still
  cites its own filing and page.
- **Several filings, one answer.** When a question names no company, every filing is
  searched. The prompt then requires the model to answer for each filing separately under
  its company name and never merge figures across filings. Each excerpt's Source line
  names its company, for filings ingested after this change.

Answers drawn from one filing say so under the answer ("Answered from acme-10k.pdf only.").

---

## Quickstart

**Requirements:** Python 3.10-3.12 and an OpenAI API key.

> CI tests 3.11 and 3.12. The optional RAGAs evaluation (`evaluate.py`,
> `requirements-ragas.txt`) needs 3.12 or older: `ragas` 0.2.x fails on 3.14's
> asyncio changes.

```bash
git clone https://github.com/dasosis/financial-rag.git
cd financial-rag

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env and set your real key
```

Get the sample filings: Microsoft's FY2024 and Apple's FY2025 10-Ks, as EDGAR serves them
(HTML with inline XBRL). Set `SEC_USER_AGENT` in `.env` first (the SEC asks every
automated request for a contact address), then:

```bash
python -m financial_rag.ingest --fetch-samples   # saves data/microsoft-sec-10k.htm and data/apple-sec-10k.htm, then ingests
```

The examples and benchmarks below use these two. Any other 10-K works too: fetch one
by ticker from the UI sidebar, upload a PDF, or drop a PDF or `.htm` filing into `data/`.

Then boot everything with one command:

```bash
python start.py                    # add --reingest to rebuild the vector store
```

This verifies your `.env`, ingests the filings in `data/` if `chroma_db/` is missing, starts the API on
`:8000` and the UI on `:8501`, and opens the browser once both are ready. `Ctrl-C` stops
both. Without login configured (next section) it runs in **local mode**: no sign-in, and
everything belongs to a single "local" user.

### Sign-in and per-user data

Each user signs in and sees only their own uploads, plus **sample filings** that every
user can query. Filings placed directly in `data/` are samples; uploads go to
`data/users/<user-id>/` and are visible only to the user who uploaded them. A sample
removed from `data/` is removed from the store the next time ingest runs.

Login uses Streamlit's built-in OIDC support (`st.login`) with an
[Amazon Cognito](https://aws.amazon.com/cognito/) user pool, whose hosted page handles
sign-up, email verification and password reset (free up to 10,000 monthly users). To
enable it, create a user pool with an app client that has a client secret, the
authorization-code flow, the `openid email profile` scopes, and your callback URL
(`http://localhost:8501/oauth2callback` locally). Then copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in. That file
is gitignored and never copied into the Docker image.

**The API verifies who is calling.** The UI sends the signed-in user's Cognito ID token
(`expose_tokens = "id"` in `secrets.toml`) as `Authorization: Bearer …`. `auth.py`
checks four things:
- the signature, against the user pool's published keys;
- the issuer and audience (your pool and app client);
- the expiry;
- that it is an ID token.

The user id is the token's `sub`, so a caller can't claim to be someone else, even if
the API port were exposed by mistake. Set `COGNITO_ISSUER` and `COGNITO_CLIENT_ID` in
`.env` for the API (`start.py` reads them from `secrets.toml`). Cognito ID tokens last an
hour; after that the UI offers a one-click "Sign in again".

The app fails closed: if `secrets.toml` has no `[auth]` section, it refuses to start
unless `AUTH_DISABLED=1` is set, which `start.py` does for you in local mode. The API
fails closed too. It trusts an `X-User-Id` header only with `AUTH_DISABLED=1`, and with
neither that nor the Cognito settings it refuses every request.

### Uploading your own filings

Use **Upload and index** in the UI sidebar (several PDFs at once is fine), or call the API
directly. Each upload is chunked and embedded immediately, saved to your folder under
`data/users/`, and becomes queryable as soon as the request returns; the sidebar lists
your filings and the shared samples. Uploading a file with the same name as one of your
own **replaces** it rather than duplicating its passages; other users' files are never
affected. Scanned PDFs without a text layer are rejected — run OCR first.

### Usage limits

Every question makes up to four gpt-4o-mini calls and every upload embeds a whole filing,
so on a public site anyone who signs up spends your OpenAI money. `usage.py` caps it:

| Limit | Default | Environment variable |
| --- | --- | --- |
| Questions per user per month | 50 | `QUESTIONS_PER_MONTH` |
| Uploads per user per month | 3 | `UPLOADS_PER_MONTH` |
| Questions per user per minute | 5 | `QUESTIONS_PER_MINUTE` |
| Questions per day, all users together | 500 | `QUESTIONS_PER_DAY_ALL_USERS` |
| Pages per uploaded PDF | 500 | `MAX_UPLOAD_PAGES` in `api.py` |

Months and days are UTC calendar periods. Set a variable to `0` to turn that limit off;
the API reads them from `.env`. Sign-up is free, so per-user limits alone don't bound
the bill: the daily cap does. A question costs about $0.001 on average and rarely more
than $0.002, so 500 a day comes to at most about $30 a month. Set a monthly spending cap
in your OpenAI account as well.

Requests that fail (an API error, a PDF that won't ingest) are refunded. A request over
a limit gets `429` with a `Retry-After` header and a message the UI shows as is; the
sidebar shows the user's counts for the month. The `local` user, which exists only when
login is disabled, is never limited. Counts are kept in `chroma_db/usage.sqlite3`.

### Running the pieces individually

```bash
python -m financial_rag.ingest                          # build the vector store
uvicorn financial_rag.api:app --reload --port 8000      # API
streamlit run financial_rag/app.py                      # UI (needs the API running)
python -m financial_rag.rag "What was total revenue?"   # one-shot CLI query
```

---

## Run with Docker

Requires Docker with the Compose plugin, and a `.env` file with your `OPENAI_API_KEY`
(the key reaches the containers via `env_file` and is never baked into the image).

```bash
docker compose run --rm api python -m financial_rag.ingest --fetch-samples   # samples, once
docker compose run --rm ingest     # later re-ingests: embed the filings in data/
docker compose up                  # app on http://localhost via the Caddy proxy
```

Only the **Caddy** proxy is reachable from outside the host (ports 80/443). Set
`SITE_ADDRESS` in `.env` to a public hostname and Caddy gets an HTTPS certificate for it
automatically. No domain? A free hostname such as `3-106-84-6.sslip.io` (your server's IP
with dashes) resolves to that IP and works too. The API and UI are published on
`127.0.0.1` only, for local debugging. Login config is mounted read-only from
`./.streamlit`; for local Docker runs without Cognito, set `AUTH_DISABLED=1` in `.env`.

One image serves all three roles — API, UI, and the ingest job — selected by the compose
`command`. The vector store lives in the named volume `chroma-data`, so it survives
rebuilds and restarts; re-run the ingest job to rebuild it after changing chunking. `./data`
is bind-mounted into the API (writable, so UI uploads land there) and the ingest job
(read-only), so re-running ingest rebuilds the store from both your own and uploaded PDFs.

Details worth knowing:

- The image installs **CPU-only torch** before the rest of the requirements, keeping
  ~2 GB of CUDA wheels out of the build.
- The **reranker model is baked into the image** at build time, so no Hugging Face
  download happens at runtime — the running containers only need network access to
  OpenAI.
- The UI reaches the API at `http://api:8000` via the compose network; `app.py` reads
  this from the `API_BASE_URL` environment variable and falls back to `localhost` outside
  Docker.
- Containers run as a non-root user, and the API exposes a `/health` healthcheck that
  gates UI startup.

---

## LLM gateway

`gateway/` is an optional service in Go between the API and the model providers. It
speaks the OpenAI API, so the app only changes where it sends its calls: set
`LLM_GATEWAY_URL` and every chat and embeddings call goes to the gateway instead of
OpenAI. Without it the app calls OpenAI directly, exactly as before.

```
   api.py ──► llm-gateway (Caddy, round robin) ─┬─► llm-gateway-1 ─┐   ┌─► OpenAI
                                                ├─► llm-gateway-2 ─┼───┤
                                                └─► llm-gateway-3 ─┘   └─► OpenRouter (backup)
                                                       │    │
                                    limits & budgets (Redis)  response cache (Redis)
```

Each call is checked and recorded in this order:
- **Who's calling.** The gateway's own key (`LLM_GATEWAY_KEY`) and an `X-Consumer-Id`,
  either `user:<id>` or `system:ingest` / `system:eval`. Without both it answers `401`.
  The OpenAI key then lives only in the gateway.
- **Personal data is redacted** before anything else sees the prompt: emails,
  formatted phone numbers, card numbers (Luhn-checked), dashed SSNs, PAN and Aadhaar
  (Verhoeff-checked) become `[EMAIL]`, `[PHONE]` and so on. A test runs the redactor
  over both sample 10-Ks and every benchmark question and fails on any changed
  figure, year, EPS value, share count or citation. Only counts are recorded.
- **Exact repeats come from a cache**: temperature-0 calls, keyed by a hash of every
  field that can change the answer, never by similarity. "Revenue in 2023" and "in
  2024" embed almost identically, and a semantic cache would answer one with the
  other's figures. Only answers that finished normally are stored, for 7 days, and
  the benchmarks bypass the cache (`Cache-Control: no-cache`).
- **Rate limits and daily budgets** are shared by the replicas through Redis, checked
  and taken in one Lua script that reads Redis's clock:
  - 60 calls a minute per user and 300 for everyone;
  - 320,000 weighted tokens a day per user, about 20 questions at the ~16,000 tokens
    one uses. A gpt-4o-mini input token counts 1, an output token 4.

  Each call reserves an estimate and is corrected to its real usage afterwards. Over
  a limit, the app answers its user with the usual `429` (detail and `Retry-After`)
  and refunds the question. If that Redis is down, each replica keeps a third of
  the limits itself and says so in a metric.
- **Fallback.** OpenAI first, then OpenRouter's `openai/gpt-4o-mini`, with a circuit
  breaker per provider. Nothing switches once the first byte has gone back to the
  app. Embeddings never fail over to a different model, because its vectors
  couldn't be compared with the stored ones; with OpenAI down they get `503`. A
  backup's structured output goes out only if it matches the schema. When no
  provider can answer, the planner, reranker and follow-up rewrite fall back as
  they do on any failed call, and the answer itself is a `503`.

The details, and every setting, are in [gateway/README.md](gateway/README.md).

### Running it

```bash
# .env (the API's):
#   LLM_GATEWAY_URL=http://llm-gateway:8080/v1
#   LLM_GATEWAY_KEY=<a long random string>
# .env.gateway (read by the gateway only):
#   OPENAI_API_KEY=sk-...
#   OPENROUTER_API_KEY=sk-or-...        # optional: the backup
docker compose --profile gateway up -d                       # 3 replicas, balancer, 2 Redis
docker compose --profile gateway --profile metrics up -d     # and Prometheus + Grafana
```

`docker compose up` without a profile starts exactly what it did before. None of the
gateway's services are published to the host. Grafana, like Jaeger, listens on the
host's loopback only: `ssh -L 3000:localhost:3000 <host>`, then
http://localhost:3000, user `admin`. It needs `GRAFANA_ADMIN_PASSWORD` in `.env` and
won't start without one; Grafana keeps the password it was first started with.
Anonymous access is off. Its one dashboard is provisioned from
`gateway/observability`:
- request rate and errors by code;
- latency and time to first byte (p50 / p95 / p99);
- cache hit ratio, cost and cost saved, and tokens by model;
- which provider served each call, and each replica's breakers;
- the limiter's state, limit and budget rejections, and redactions.

No metric carries a user id.

Every container has a `mem_limit`, sized for a 2 GB host:

| Service | Limit | Used under load |
| --- | --- | --- |
| 3 gateway replicas | 3 × 40 MB | ~11 MB each |
| Load balancer (Caddy) | 40 MB | ~20 MB |
| Limits Redis (`noeviction`) | 32 MB | ~4 MB |
| Cache Redis (`allkeys-lru`, 64 MB of entries) | 96 MB | ~15 MB |
| Prometheus | 128 MB | ~25 MB |
| Grafana | 160 MB | ~80 MB |
| **Total** | **576 MB** | **~180 MB** |

### Measured

`eval_answers.py` through the gateway on the real models. The eval process had no
OpenAI key, so every call went through the gateway:

| | Correct | Route | Verified | Cited | Cost |
| --- | --- | --- | --- | --- | --- |
| Passthrough | **52/52** | **52/52** | **52/52** | **52/52** | $0.049 |
| Everything on: redaction, limits, cache (bypassed), breakers | **51/52** | **52/52** | **52/52** | **52/52** | $0.048 |
| Forced onto the backup (`X-Force-Provider: openrouter`) | not run yet: needs an `OPENROUTER_API_KEY` | | | | |
| Direct to OpenAI (nightly, last five runs) | 51–52/52 | 51–52/52 | 52/52 | 51–52/52 | ~$0.05 |

The one miss with everything on is the goodwill question taking a segment's goodwill
(51,235) from the right table for the total (50,969), one of the text answers that
vary between runs. Nothing on that page was redacted: the run redacted two phone
numbers, both on cover pages.

A load test (`gateway/loadtest`) against the fake upstream (`gateway/cmd/fakeupstream`,
which answers after 20 ms) with 20 concurrent clients. It ran through the balancer
and three replicas, with redaction, both Redis and the breakers on, under Docker
Desktop on a laptop:

| | p50 | p99 |
| --- | --- | --- |
| Straight to the fake | 21.5 ms | 24 ms |
| Through the gateway | 25 ms | 38–53 ms |
| **The gateway's overhead** | **+3–4 ms** | **+14–29 ms** |
| Time to first token, streamed, through the gateway | 28–29 ms | 56–78 ms |
| A cache hit | 8–9.5 ms | 29–36 ms |

The global limit across the three replicas, at 600 calls a minute with a burst of
60, with 50 clients calling as fast as they could:

| Run | Most the limit allows (burst + rate × time) | Admitted |
| --- | --- | --- |
| 3 × 20 s | 260.1–260.2 | 259, 259, 259 (99.6%) |
| 60 s | 660.2 | 659 (99.8%) |

It never let more through than the limit allows. A real model takes seconds to
answer, so the overhead is a few milliseconds in thousands.

### Tests

```bash
cd gateway && go vet ./... && go test -race ./...     # CI's Go (gateway) job, with a real Redis
RUN_GATEWAY_TESTS=1 pytest tests/test_gateway_live.py   # the app's real models through it
```

The Go tests run the gateway against an in-process fake upstream (SSE included) and
miniredis. They cover:
- the global limit across three replicas and concurrent clients, which is never
  exceeded, plus one test on a real Redis;
- budgets reserved and corrected, and the per-replica fallback;
- cache keys, storage rules and the SSE replay;
- breakers, failover before the first byte only, and embeddings refusing another model;
- the backup's structured output checked against its schema;
- the redactor over the samples.

The Python tests need neither Go nor Redis. They play the gateway's responses with a
mock transport: the model factory, consumer ids, and refusals arriving as the API's
`429` and `503` with a refund. The opt-in live test builds the gateway and runs the
app's real `ChatOpenAI` and `OpenAIEmbeddings` through it: invoke, stream,
`with_structured_output`, embeddings, a spent budget, the cache and the backup.

---

## API

Every endpoint except `/health` requires the signed-in user's Cognito ID token,
`Authorization: Bearer <token>` (see [Sign-in and per-user data](#sign-in-and-per-user-data)).
A missing, forged, expired or wrong-audience token gets `401`, and a server without the
Cognito settings answers `503`. In local mode (`AUTH_DISABLED=1`, as in the examples
below), an `X-User-Id` header names the user instead (`[A-Za-z0-9-]`, up to 64
characters). Keep port 8000 private anyway: only the UI needs to reach it.

`POST /query`: answers from the caller's filings and the shared samples

```bash
curl -X POST http://localhost:8000/query \
  -H "X-User-Id: local" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the current ratio?"}'
```

`route` says which engine answered. For `"calculation"`, `calculations` holds each result
as an exact decimal (percentages and multiples as fractions) with its formula and every
input figure, and `sources` holds one citation per input row:

```json
{
  "answer": "**Current ratio** · microsoft-sec-10k.htm\n- **FY2024: 1.27x** = Total current assets ÷ Total current liabilities, from: …",
  "route": "calculation",
  "calculations": [
    {
      "metric": "current_ratio", "name": "Current ratio",
      "source_file": "microsoft-sec-10k.htm", "fiscal_year": 2024,
      "value": "1.274954903181520680682598215", "display": "1.27x",
      "formula": "Total current assets ÷ Total current liabilities", "problem": null,
      "inputs": [
        {
          "metric": "total_current_assets", "name": "Total current assets",
          "fiscal_year": 2024, "value": "159734000000", "unit": "USD",
          "display": "$159,734 million", "label": "Total current assets",
          "source_file": "microsoft-sec-10k.htm", "page": 58,
          "line": "Total current assets 159,734 184,257", "shared": true
        },
        { "metric": "total_current_liabilities", "…": "…" }
      ]
    }
  ],
  "sources": [
    { "source_file": "microsoft-sec-10k.htm", "page": 58,
      "excerpt": "Total current assets 159,734 184,257" },
    { "source_file": "microsoft-sec-10k.htm", "page": 58,
      "excerpt": "Total current liabilities 125,286 104,149" }
  ]
}
```

Other questions return `"route": "text"`, an empty `calculations` list, and the
retrieved passages as `sources`. Each of those carries a `retrieval` record of how its
chunk was found; see [Hybrid retrieval](#hybrid-retrieval).

Add `"history": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]`
(earlier turns, oldest first) so a follow-up such as "And in 2023?" is read in context.
`question` in the response is the question as answered, rewritten if it was a follow-up;
see [Follow-up questions](#follow-up-questions).

Add `"source_file": "acme-10k.pdf"` to the request to answer from that one filing. Without
it, a question that names a company is narrowed to that company's filing. `filing` in the
response names the one filing an answer was drawn from, or is `null` if it drew on all of
them; see [Which filing a question is about](#which-filing-a-question-is-about).

Status codes: `422` empty question · `404` `source_file` isn't one of the caller's
filings or a sample · `503` vector store missing (run `ingest.py` or upload a PDF) ·
`500` inference error. `GET /health` returns `{"status": "ok"}`.

`POST /query/stream` takes the same request and answers as it works, in newline-delimited
JSON. The UI uses it, so a text answer shows each step and then its words as they're
written, instead of seven silent seconds:

```
{"type": "step", "step": "Reading the question"}
{"type": "step", "step": "Searching the filings"}
{"type": "step", "step": "Ranking the passages"}
{"type": "step", "step": "Writing the answer"}
{"type": "token", "text": "Apple's independent auditor is"}
…
{"type": "step", "step": "Checking its figures against the filing"}
{"type": "done", "response": {"answer": "…", "sources": […], …}}
```

- **Steps** come from the tracing spans as they start (`tracing.listening`), so they always
  match what the pipeline is doing.
- **Text is a preview.** If the check against the excerpts fails, a `{"type": "retry"}`
  withdraws it and a corrected answer follows. `done` carries the checked answer, exactly as
  `POST /query` returns it, and that's what the UI keeps.
- **Errors after the stream starts** arrive as `{"type": "error", "status": 404, "detail":
  "…"}`, and the question isn't counted. An empty question (`422`) and a usage limit
  (`429`) are refused before anything streams.

`POST /documents` — upload a PDF (multipart field `file`, max 50 MB)

```bash
curl -X POST http://localhost:8000/documents -H "X-User-Id: local" -F "file=@microsoft-sec-10k.pdf"
```

```json
{"source_file": "microsoft-sec-10k.pdf", "chunks": 155, "figures": 59, "shared": false,
 "company": "Microsoft Corporation"}
```

The filename is reduced to a safe basename (`[A-Za-z0-9._-]`), which is what citations
show. Status codes: `201` indexed · `413` too large · `415` not a PDF · `422` no
extractable text, unreadable PDF or over 500 pages · `429` monthly upload limit reached ·
`500` ingestion error.

`GET /documents` lists the caller's files and the shared samples (`"shared": true`), with
the number of structured figures extracted from each and the company named on its cover
page (`null` if none was found):
`{"documents": [{"source_file": "…", "chunks": 155, "figures": 59, "shared": false, "company": "…"}]}`.

`GET /financials` (optional `?source_file=`) lists those figures, each with its value,
fiscal year, page and the row as printed: `{"figures": [{"metric": "revenue", …}]}`.

`POST /filings` with `{"ticker": "AAPL"}` fetches the company's latest 10-K from SEC EDGAR
(add `"fiscal_year": 2024` for an earlier one; EDGAR lists about ten years) as a
background job: `202` with `{"job_id": …, "state": "queued", …}`. It answers `404` for
an unknown ticker, `422` for a malformed one, `429` over the upload limit, and `503` when
EDGAR can't be reached or `SEC_USER_AGENT` isn't set. `GET /jobs/{job_id}` reports progress:
`state` is queued, running, done or failed, `step` says what a running job is doing,
`error` says why a job failed, and `document` describes the filing once it's done. A job
is only visible to the user who started it.

`GET /usage` returns the caller's counts for this month and the limits they count
against (`0` = unlimited), with the date the counts reset:
`{"questions": 12, "question_limit": 50, "uploads": 1, "upload_limit": 3, "resets": "2026-11-01"}`.
`POST /query` answers `429` once a question limit is reached (see
[Usage limits](#usage-limits)).

---

## Configuration

Retrieval behaviour is tuned by the constants at the top of `rag.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `RETRIEVE_K` | `20` | Semantic candidates pulled from ChromaDB before filtering. |
| `SCORE_THRESHOLD` | `0.10` | Minimum semantic relevance score; below this a semantic hit is discarded. |
| `KEYWORD_K` | `20` | BM25 candidates. |
| `RERANK_CANDIDATES` | `30` | Fused candidates the cross-encoder scores. |
| `TOP_K` | `5` | Chunks sent to the LLM after reranking. |
| `HYBRID_SEARCH` | env, `1` | Set the `HYBRID_SEARCH` environment variable to `0` for semantic-only retrieval. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for the first reranking stage. |
| `RERANKER` | env, `llm` | `llm` has gpt-4o-mini order the cross-encoder's top candidates; `cross-encoder` skips that call. |
| `LLM_RERANK_CANDIDATES` | `15` | How many of the cross-encoder's top candidates the LLM orders. |

BM25 parameters, fusion weights and the tokenizer live in `hybrid.py`.
Chunking is configured in `ingest.py` (`CHUNK_SIZE`, `CHUNK_OVERLAP`). Changing either
requires a re-ingest: `python start.py --reingest`.

Model names live in `config.py`. Every language-model call uses `CHAT_MODEL`
(`gpt-4o-mini` by default), which the environment variable of the same name overrides.
To compare a model, run the benchmark with it:
`CHAT_MODEL=<model> python -m financial_rag.eval_answers`. Add the model's prices to `PRICES` in
`eval_answers.py` for a cost estimate. The embedding model is fixed, because changing it
means re-ingesting.

The only required environment variable is `OPENAI_API_KEY`, read from `.env`. The usage
limits are optional variables in the same file (see [Usage limits](#usage-limits)).

### Tracing

Every question gets an OpenTelemetry trace (`tracing.py`), with one span per step:
`answer`, then `rewrite_follow_up`, `plan`, and either `calculate` or `text_answer`. A
text answer breaks down further into `retrieve` (`search.semantic`, `search.keyword`,
`rerank.cross_encoder`, `rerank.llm`), `generate` and `check`. Uploads and fetches are
traced too: `fetch_10k`, `ingest`, `extract_figures` and `xbrl_check`.

Tracing is off unless configured, and the traces stay on your own machines:
- `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318` in `.env` sends traces over OTLP/HTTP
  to a collector you run. `compose.yaml` has one: `docker compose --profile tracing up
  -d` starts Jaeger, capped at 300 MB of memory, with its UI on the host's loopback
  only. On a server, reach it with `ssh -L 16686:localhost:16686 <host>`, then
  http://localhost:16686.
- `TRACE_CONSOLE=1` prints the spans to the log instead.

Spans record timings, routes, metric keys, counts and file names. They never record
the question, the answer or who asked.

`eval_answers.py` collects its spans in memory and reports the median time per step:

```
median time per step: answer 1.08 s (x51), plan 0.96 s (x51), calculate 0.00 s (x35),
text_answer 6.74 s (x16), retrieve 5.63 s (x17), search.semantic 0.35 s (x17),
rerank.cross_encoder 4.15 s (x17), rerank.llm 1.02 s (x17), generate 1.15 s (x16), …
```

On a laptop CPU, the cross-encoder scoring 30 candidates is most of a text answer's
time.

---

## Evaluation

Three evaluations, each answering a different question:

| Script | Measures | Cost per run |
| --- | --- | --- |
| `eval_answers.py` | Whether final answers are right, end to end, across every route | ~$0.03 |
| `eval_retrieval.py` | Whether search puts the right passage in front of the model | ~$0.15 |
| `eval_xbrl.py` | Whether calculated answers match the SEC's XBRL values, for any list of companies | ~$0.03 per new company |
| `evaluate.py` (optional) | RAGAs faithfulness and relevancy of text answers, judged by an LLM | 28 answers plus RAGAs judge calls (not measured) |

`eval_answers.py` is the one to run after a change. It scores every route against
figures checked by hand. `evaluate.py` came first: it judges only text answers, and it
marks a correct "not found in context" as unfaithful. Its packages are kept out of
`requirements.txt` and the Docker image; install `requirements-ragas.txt` to run it.

### Nightly runs

`.github/workflows/nightly-eval.yml` runs the benchmarks against the real models, so a
regression from a dependency, a model update or a merge shows up the next morning:
- **Every night** (00:00 IST): `eval_answers.py` on the two samples, about $0.08.
- **Sundays**: `eval_xbrl.py` on its ten companies as well, about $0.40.

Each step has a hard spending cap and a minimum score, and the run fails if either is
hit (GitHub emails the person who last edited the schedule):

```bash
python -m financial_rag.eval_answers --max-cost 0.25 --min-score 0.94 --summary summary.md
```

- `--max-cost` stops before the estimated chat-model spend passes the cap (exit 3).
  Embeddings, a few cents a run, aren't counted. A model with no price in `PRICES` is
  refused when a cap is set.
- `--min-score` is the share of questions that must fully pass: correct, routed,
  verified and cited (exit 1 below it). 0.94 allows for the two text questions that
  vary between runs.
- `--summary` appends a Markdown table per group, the median time per step and the
  questions that didn't pass. The workflow writes it to the run's summary page, and
  keeps the full results as an artifact for two weeks.

The workflow needs two repository secrets: `OPENAI_API_KEY` and `SEC_USER_AGENT`. It
doesn't run on forks, and it can also be started by hand from the Actions tab, with
the XBRL benchmark as an option.

### End-to-end answers

```bash
python -m financial_rag.eval_answers
```

Runs 52 questions through `router.answer` exactly as the API does: follow-up rewriting,
planning, filing choice, calculation or retrieval, reranking and the answer check. It
scores the final answers against figures checked by hand in two filings, Microsoft's
FY2024 10-K and Apple's FY2025 10-K, both loaded, so a question that names no company is
answered with both in play:
- **correct:** every expected figure is in the answer, and nothing that mustn't be;
- **route:** the expected engine answered;
- **verified:** no "Not verified" warning;
- **cited:** the answer cites a page.

The groups:
- 10 calculations;
- 5 on parts of the company: segment revenue and operating margin, US revenue, product
  revenue, and growth;
- 10 text lookups, including "Microsoft Cloud revenue" (not a breakdown part) and a
  "dividends declared" trap;
- 4 follow-ups with conversation history;
- 16 on Apple: 9 calculations (including diluted share count, which tests the share
  scale), 3 on product and country revenue, 3 text lookups, and a follow-up;
- 5 comparisons of Apple and Microsoft: two calculations, their auditors (text, from each
  filing), how their AI risks differ (a summary of each filing, cited, where the old prompt
  answered "not found in context"), and one naming Tesla, which has no filing (answered for
  Microsoft, saying so);
- 2 that should be refused: Tesla's revenue (no filing) and Microsoft's FY2030 revenue.

It also reports latency, gpt-4o-mini tokens and an estimated cost per question.

| | Correct | Route | Verified | Cited |
| --- | --- | --- | --- | --- |
| First run (27 questions) | 22/27 | 24/27 | 27/27 | 27/27 |
| After the fixes below (two runs) | 27/27 | 27/27 | 27/27 | 27/27 |
| With breakdowns, 31 questions (two runs) | **31/31** | **31/31** | **31/31** | **31/31** |
| With Apple, 47 questions (two runs) | **46/47** | **47/47** | **47/47** | **46/47** |
| HTML samples and comparisons, 51 questions | **51/51** | **51/51** | **51/51** | **51/51** |
| With a qualitative comparison, 52 questions (two runs) | **52/52** | **52/52** | **52/52** | **52/52** |

Adding Apple's 10-K exposed four problems that one company had hidden:
- **No figures at all.** Apple's statements put each column's date on two lines
  ("September 27," then "2025"), so no header line held two years and every mapping
  failed verification. Year headers are now read either way.
- **Share counts 1,000× too large, once that was fixed.** Apple reports shares in
  thousands while dollars are in millions. The scale note's wording for shares is now
  read separately.
- **"United States revenue" answered from text.** Apple prints the country as "U.S.";
  "U.S.", "US" and "USA" now match "United States".
- **Citing Apple's printed page numbers.** The model cited "p.22" from the footer
  "Apple Inc. | 2025 Form 10-K | 22" instead of PDF page 25. The answer check flagged it,
  and footers like that are now removed from the excerpts the model sees.

The one miss: "short-term unearned revenue" is answered "not found in context". The row
is retrieved, but its chunk starts below the "June 30, 2024 / 2023" column header, so
the model can't tell which column is 2024. It answered correctly when only Microsoft was
loaded. With Apple loaded, the keyword statistics change and so do the other retrieved
chunks. Narrowed to Microsoft's filing, it answers correctly. Carrying a table's column
header into each of its chunks at ingest would fix it.

What the first run caught, all answered confidently as calculations:
- "Intelligent Cloud revenue", "LinkedIn revenue" and the follow-up "And Dynamics?" were
  answered with total revenue. This is fixed by [qualifiers](#financial-analysis-engine).
- "Dividends declared" was answered with dividends paid. Also fixed by qualifiers.
- "Apple's revenue" was answered with Microsoft's revenue. Now a named company with no
  filing gets "No filing for Apple is available…", the filings that are, and `"route":
  "none"`.

Fixing the third one at first broke "goodwill from the Activision Blizzard acquisition",
which was refused because Activision Blizzard is a company too. The planner now extracts
only the company whose own filing is asked about, not acquisitions, competitors or
customers it mentions.

Per question, the median is 6.9 seconds and about 16,000 input tokens, and the mean cost
is about $0.001. Most input tokens are cached by OpenAI, which is why the cost is lower
than the token count suggests.

### Any company, checked against XBRL

```bash
python -m financial_rag.eval_xbrl                  # ten companies, below
python -m financial_rag.eval_xbrl WMT JPM          # any tickers
python -m financial_rag.eval_xbrl --history 3 MSFT # five years from three 10-Ks
```

`eval_answers.py` needs every expected answer checked by hand, so it covers two filings.
This one needs none:
1. It fetches each company's latest 10-K from EDGAR into a store of its own
   (`chroma_db/eval_xbrl/`).
2. It asks for six reported figures and four derived ones: operating and net margin, free
   cash flow, and current ratio.
3. Each expected answer is the company's own XBRL value from that filing, formatted the way
   the engine formats it. Derived metrics use the engine's formulas with XBRL inputs.

All the companies share one store, so naming the company in each question also tests
choosing the right filing. A question is generated only when XBRL has every value it needs.

First run, six companies, 58 questions:

| | Correct | Route | Cited |
| --- | --- | --- | --- |
| Apple, Microsoft (FY2026), NVIDIA, Alphabet | 40/40 | 40/40 | 40/40 |
| Coca-Cola | 7/10 | 7/10 | 8/10 |
| Johnson & Johnson | 7/8 | 6/8 | 7/8 |
| **All** | **54/58** | **53/58** | **55/58** |

NVIDIA and Alphabet had never been tested. The misses are real extraction gaps, and the XBRL
check is why none of them became a wrong number:
- **Coca-Cola net income.** Coca-Cola prints "Consolidated Net Income" (13,137) above "Net
  Income Attributable to Shareowners" (13,107). The extraction took the consolidated line,
  which includes minority owners' share. XBRL reports the attributable one, so the check
  dropped all three years, and net income and net margin weren't calculated.
- **Coca-Cola capital expenditures** weren't mapped, though the row is on the cash-flow
  statement.
- **Johnson & Johnson net income.** "Net earnings" and "Net earnings from continuing
  operations" tied in verification, so neither was used.

These were fixed (58/58). Then four harder companies, now among the defaults, scored
13/28: JPMorgan (a bank), Travelers (an insurer), Realty Income (a REIT) and Target (a
retailer whose 52/53-week year ends in January). The fixes:
- **Target's years.** Target calls the year to January 31, 2026 "fiscal 2025". Years are
  now named as each 10-K's own XBRL names them, and a balance sheet dated a year after
  the year it closes is relabelled to match.
- **Statement pages.** Target's "(millions)" scale note; a Travelers data row that names
  the income statement; Realty Income's "Statements of Income and Comprehensive Income".
- **Substituted rows.** Asked for a line item a bank or insurer doesn't report, the model
  pointed at a neighbour ("Total assets" for current assets). Each line item now has a
  pattern its printed label must fit.
- **Re-reading a contradicted figure** from the one row that prints XBRL's values, such as
  Realty Income's net income attributable to the company.
- **Banks' own line items:** net interest income, noninterest revenue and expense, the
  provision for credit losses, and the efficiency ratio.

All ten defaults now score **91/91**.

**History.** With `--history N` each company's latest N 10-Ks are indexed, and the
questions cover every year they report between them (N + 2 years of income): revenue,
cost of revenue and net income year by year, then compound revenue growth across the
span. Filings overlap, so a year's expected value is what the most recently filed of
them reports. `--history 3` on Microsoft, Apple, Target and Coca-Cola: **64/64**,
including Target's restated FY2022 cost of sales ($82,306 million, not the $82,229
million first reported).

### Answer faithfulness (RAGAs, optional)

```bash
pip install -r requirements-ragas.txt   # Python 3.12 or older
python -m financial_rag.evaluate
```

Runs [RAGAs](https://docs.ragas.io) over the 28 questions in `EVAL_QUESTIONS`, scoring
**faithfulness** (are the claims grounded in the retrieved context?) and **answer
relevancy**. Results are written to `evaluation_results.json`; if that file already exists,
the run prints a before/after table so you can see whether a retrieval change helped.

The evaluation calls the text-retrieval path (`rag.query`) directly, so it measures
retrieval and generation only. In the app, questions about the catalog's metrics are
answered by the analysis engine instead; its correctness is covered by the unit tests,
which check results against the filing's figures.

The questions span the income statement, balance sheet, cash flow statement, segment note,
and narrative sections, and their ground truths were read directly from Microsoft's FY2024
10-K. Point the pipeline at a different filing and you must replace the whole list.

### Results

Measured on Microsoft's FY2024 10-K with `gpt-4o-mini` as the judge — 28 questions, two
full evaluation passes per configuration. "Before" and "after" refer to the table-aware
chunking fix that stops prose being glued to the table above it:

| Metric | Before chunking fix | After chunking fix |
| --- | --- | --- |
| Faithfulness | 0.875 (both passes) | **0.911** (0.929 / 0.893) |
| Answer relevancy | 0.887 | 0.886 |

The faithfulness gain is driven almost entirely by one question — capital expenditures,
a cash-flow-statement line — which went from 0.50 to 1.00 consistently in both passes.
That is exactly the table-adjacent-prose case the fix targets. Relevancy is unchanged
because both configurations retrieve the same pages in the same order; only the chunk
boundaries within them differ. Individual questions can still flip between runs of an
identical configuration (the judge is an LLM), so treat mean deltas smaller than ~0.04
as noise.

Two things to know before reading the numbers:

- **A correct refusal scores 0.0 on answer relevancy.** RAGAs treats "not found in
  context" as non-committal and scores it zero, so the pipeline is penalised for exactly
  the behaviour the strict prompt is meant to produce.
- **The eval must see the same context the model saw.** `evaluate.py` builds contexts with
  `rag.doc_to_context()`, which prefixes each chunk with its source and page. Passing the
  bare `page_content` makes the page citation the prompt mandates look like an unsupported
  claim, which pins faithfulness near 0.5 for every correctly-cited answer.

### Retrieval benchmark

```bash
python -m financial_rag.eval_retrieval            # about $0.15 of gpt-4o-mini calls
python -m financial_rag.eval_retrieval --no-llm   # skip the LLM reranking configuration
```

Scores retrieval alone. A question counts as a hit when a retrieved chunk contains its
evidence: the figure or phrase from the ground truth, exactly as the filing prints it. It
compares three configurations of the pipeline on 72 questions about Microsoft's FY2024
10-K, in three sets:

- **`evaluate.py` (26):** the questions whose ground truth has a figure or name to look
  for.
- **Exact-term (10):** typed the way analysts do, e.g. "FY24 additions to property and
  equipment" or "Where does the filing report 245,122?".
- **More (36):** older fiscal years, other line items, segments, notes and management's
  discussion. These were written, and each answer checked against the filing text, before
  any configuration was scored.

| Configuration | Ranked first | In the top 5 (sent to the model) | MRR |
| --- | --- | --- | --- |
| semantic + cross-encoder (the original pipeline) | 47/72 | 65/72 | 0.750 |
| hybrid + cross-encoder | 50/72 | 68/72 | 0.790 |
| **hybrid + LLM rerank (default)** | **66/72** | **72/72** | **0.943** |

By set, hybrid + LLM rerank ranks the right passage first on 24/26, 9/10 and 33/36
questions. The cross-encoder-only configurations manage 16-17/26, 7-8/10 and 24-25/36.

What the numbers show:
- **Hybrid search fixes recall.** It finds passages that semantic search never surfaced,
  such as the employee count and an exact "245,122" lookup.
- **The LLM fixes ordering.** With the cross-encoder alone, the right chunk often reached
  the reranker but was demoted, usually behind a related note or the stockholders' equity
  statement. gpt-4o-mini, told to prefer the passage that states the exact line item and
  year, gets those right.
- **Cost.** The extra step costs one gpt-4o-mini call per question: about 11,000 input
  tokens (15 chunks), a median of 1.3 seconds, and about $0.002.

Tested and dropped:
- **`BAAI/bge-reranker-base`** instead of the cross-encoder: 34/72 ranked first, worse
  than the small cross-encoder. It also took about 27 seconds per question on CPU.
- **Section labels.** Each chunk was prefixed with its 10-K heading, e.g. "ITEM 8 ›
  CASH FLOWS STATEMENTS". That made every reranker slightly worse (cross-encoder 50 →
  44 ranked first) and needs a re-ingest, so it isn't used. The code is kept on the local
  branch `experiment/section-labels`.
- **Reranking only the part of each chunk that matched the question.** The reranker lost
  the statement heading, and MRR fell from 0.79 to 0.69.
- **Blending the reranker's order with the first-stage ranks.** No gain.
- **Tuning the fusion weights.** No measurable effect: the 30 reranked candidates cover
  nearly the whole union of the two 20-chunk searches.

It's one filing and 72 questions, so treat single-question differences as anecdotes.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover:
- the chunking logic and prompt formatting;
- fetching by ticker: the job's steps and result, the file saved under the user's folder,
  refunds and error messages for failures, unknown and malformed tickers, class-share
  tickers, EDGAR outages, the upload limit, jobs private to their owner, one job at a
  time, and jobs a restart interrupted;
- EDGAR HTML filings: page breaks, hidden XBRL headers, inline tags inside sentences, rows
  read as statement rows, column headings repeated in chunks that lost them, and HTML
  files found and ingested next to PDFs;
- the upload path: validation, staging, replace-on-reupload;
- the XBRL check: matching by tag, unit, fiscal year and full-year period (a quarter
  ending the same day never matches); contradicted figures dropped; no network, no
  contact address or an unknown company leaves figures unchanged; EDGAR caching;
- sign-in tokens: valid ID tokens accepted; forged, unsigned, expired, wrong-issuer,
  wrong-audience and access tokens rejected; `X-User-Id` ignored unless in local mode;
- usage limits: monthly, per-minute and daily caps, calendar resets, refunds of failed
  requests, `429` responses, and that an over-limit upload is never ingested;
- hybrid search: the tokenizer, BM25, query profiles, rank fusion and citation excerpts;
- answer checking: blended figures, real figures cited to the wrong page, citations of
  pages that weren't retrieved, figures quoted from the question, and the retry and
  warning;
- LLM reranking (with the model faked): the LLM's order applied to its candidates, exact
  value matches still first, fallback to the cross-encoder when the call fails, and
  parsing of the model's ranking;
- per-user isolation: user-id validation, and retrieval against a temporary ChromaDB (fake
  embeddings and reranker). Neither the semantic nor the keyword path returns another
  user's chunks, keyword hits carry their provenance, and replaced filings drop out of the
  keyword index;
- the analysis engine:
  - **extraction**: number parsing, year-column verification (wrong-year and invented
    values rejected), wrapped and mis-encoded labels, ambiguous rows, scaling and sign
    rules, page selection, and owner-scoped storage, against statement text as PyPDF
    extracts it;
  - **calculations**: exact `Decimal` results checked by hand against Microsoft's 10-K,
    formula fallbacks, missing inputs, zero denominators, YoY and CAGR, and that every
    input is cited in the answer;
  - **routing**: calculation vs. text, fallbacks, and that the planner and engine only see
    the caller's figures;
- follow-ups: the rewritten question is what gets planned and answered, no rewrite
  without history, fallback when the rewrite fails, the transcript's limits, and the
  API's `history` and `question` fields (malformed history is rejected);
- which filing a question is about: company names from cover pages, matching a named
  company to a filing, narrowing both routes to it, the picked filing overriding the
  question, own files shadowing samples, and refusing filings the caller can't see.

None need an API key or network access. The planner's routing on real questions is an
opt-in test that calls OpenAI:

```bash
RUN_LIVE_TESTS=1 pytest tests/test_planner_live.py
```

CI (`.github/workflows/ci.yml`) runs ruff and the tests on Python 3.11 and 3.12 for every
pull request, and `main` requires those checks to pass before anything merges. Pull
requests from the repo owner and Dependabot merge automatically once they're green
(`.github/workflows/automerge.yml`); open one as a draft, or with the `no-automerge`
label, to keep it open. PRs that change workflow files are left to merge by hand.

---

## Notes and limitations

- Costs money: ingestion embeds every chunk and makes one extraction call per filing.
  Each question makes a small planning call, plus a rewrite call for follow-ups. Text
  questions also make an LLM reranking
  call (about 11,000 input tokens), the answer call, and a retry if the answer check
  fails. All of these use `gpt-4o-mini`.
- Page numbers in citations count from 1, as PDF viewers and the filings' printed page
  footers do ("p.67" is the 67th page of the PDF). They're stored as 0-based indexes and
  converted for display (`financials.page_number`), so no re-ingest is needed.
- The analysis engine reads the three primary statements, plus breakdowns of revenue and
  operating income that add up to the company totals. Other figures in the notes or
  MD&A, such as Microsoft Cloud revenue or debt maturities, are answered by text
  retrieval, which won't calculate. Amounts are assumed to be in US dollars, as in SEC
  filings, and a page must state its scale ("in millions") and a year header for its
  figures to be used.
- Every upload and question spends your OpenAI credits. [Usage limits](#usage-limits)
  cap them per user and per day; set a monthly spending cap in your OpenAI account as
  well.
- Cognito's built-in email sender only allows a limited number of emails per day
  (sign-up codes and password resets). That's fine for a demo; connect Amazon SES for
  more.
- The vector store (`chroma_db/`) and source PDFs are gitignored — both are rebuilt
  locally from your own filings.
- The reranker downloads a ~90 MB model from Hugging Face on first use.
- Paths are relative, so run the scripts from the repository root.
- Answers are generated from a limited set of retrieved excerpts. This is a demonstration
  project, not financial advice — verify any figure against the filing before relying on it.

---

## License

[MIT](LICENSE)
