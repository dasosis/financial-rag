# 📊 Financial Report Q&A Engine

A retrieval-augmented generation (RAG) system for asking grounded questions about SEC
10-K filings. It ingests PDF filings, retrieves the passages that actually answer your
question, and returns an answer with page-level citations — refusing to guess when the
figure isn't in the documents.

Built with LangChain, ChromaDB, OpenAI (`gpt-4o-mini` + `text-embedding-3-small`),
a cross-encoder reranker, FastAPI and Streamlit.

---

## Why it doesn't hallucinate numbers

Financial Q&A fails in a specific way: the model quietly derives a figure that never
appears in the filing. Three parts of the pipeline are aimed at that.

**Table-aware chunking.** 10-K financials are tables that a naive splitter cuts in half,
stranding a number from its row label. `ingest.py` detects contiguous table-like lines
(dollar amounts, comma-grouped numbers, dash separators) and hides their newlines behind
a word-joiner placeholder so the token splitter treats each table block as one unit,
then restores them after splitting.

**Threshold filtering + reranking.** Retrieval pulls 20 candidates, drops anything below
a relevance floor, reranks the survivors with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and
sends only the top 5 to the LLM. If nothing clears the floor, the API says so instead of
answering from parametric memory.

**A prompt that forbids arithmetic.** The system prompt allows only figures that appear
verbatim in the retrieved excerpts — no computing, deriving, or combining rows — and
requires a `(file.pdf, p.NN)` citation after each claim. Anything missing comes back as
"not found in context".

---

## Architecture

```
                  ingest.py
   data/*.pdf ──► load ─► table-aware split ─► embed ─► chroma_db/
                                                            │
                                                            ▼
   question ──► similarity search (k=20) ─► score filter ─► cross-encoder rerank (top 5)
                                                            │
                                                            ▼
                                              gpt-4o-mini + strict prompt
                                                            │
                                                            ▼
                                             answer + page citations
                                                            │
                          api.py (FastAPI :8000) ◄──────────┘
                                    ▲
                          app.py (Streamlit :8501)
```

| File | Role |
| --- | --- |
| `ingest.py` | Loads PDFs from `data/`, chunks them (1024 tokens / 128 overlap, table-aware), embeds and persists to ChromaDB. |
| `rag.py` | Retrieval, score-threshold filtering, cross-encoder reranking, prompting. Exposes `query()`. |
| `api.py` | FastAPI service — `POST /query`, `GET /health`. |
| `app.py` | Streamlit chat UI with expandable source citations. |
| `evaluate.py` | RAGAs evaluation (faithfulness, answer relevancy) with a before/after comparison table. |
| `start.py` | One-command boot: checks config, ingests if needed, starts both servers, opens the browser. |

---

## Quickstart

**Requirements:** Python 3.10-3.12 and an OpenAI API key.

> Python 3.13+ is not supported yet: `ragas` 0.2.x fails on 3.14's asyncio
> changes, and one of its dependencies has no wheel for it. The rest of the
> pipeline (ingest, retrieval, API, UI) runs fine on 3.14 - only `evaluate.py`
> is affected.

```bash
git clone https://github.com/dasosis/financial-rag.git
cd financial-rag

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env and set your real key
```

Add at least one 10-K PDF to `data/`. Filings are free from the SEC:
[EDGAR full-text search](https://efts.sec.gov/LATEST/search-index?q=10-K) — pick a
company, open its 10-K, and save the PDF. The examples below use Microsoft's, saved as
`data/microsoft-sec-10k.pdf`.

Then boot everything with one command:

```bash
python start.py                    # add --reingest to rebuild the vector store
```

This verifies your `.env`, ingests the PDFs if `chroma_db/` is missing, starts the API on
`:8000` and the UI on `:8501`, and opens the browser once both are ready. `Ctrl-C` stops
both.

### Running the pieces individually

```bash
python ingest.py                              # build the vector store
uvicorn api:app --reload --port 8000          # API
streamlit run app.py                          # UI (needs the API running)
python rag.py "What was total revenue?"       # one-shot CLI query
```

---

## Run with Docker

Requires Docker with the Compose plugin, and a `.env` file with your `OPENAI_API_KEY`
(the key reaches the containers via `env_file` and is never baked into the image).

```bash
docker compose run --rm ingest     # one-off: embed data/*.pdf into the store
docker compose up                  # API on :8000, UI on :8501
```

One image serves all three roles — API, UI, and the ingest job — selected by the compose
`command`. The vector store lives in the named volume `chroma-data`, so it survives
rebuilds and restarts; re-run the ingest job to rebuild it after changing chunking. PDFs
are bind-mounted read-only from `./data` into the ingest container.

Details worth knowing:

- The image installs **CPU-only torch** before the rest of the requirements, keeping
  ~2 GB of CUDA wheels out of the build.
- The **reranker model is baked into the image** at build time, so no Hugging Face
  download happens at runtime — the running containers only need network access to
  OpenAI.
- The UI reaches the API at `http://api:8000` via the compose network; `app.py` reads
  this from the `API_URL` environment variable and falls back to `localhost` outside
  Docker.
- Containers run as a non-root user, and the API exposes a `/health` healthcheck that
  gates UI startup.

---

## API

`POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was total revenue for fiscal 2024?"}'
```

```json
{
  "answer": "Total revenue was $245,122 million (microsoft-sec-10k.pdf, p.66).",
  "sources": [
    {
      "source_file": "microsoft-sec-10k.pdf",
      "page": 66,
      "excerpt": "Revenue: Product $64,773 Service and other $180,349 Total $245,122 …"
    }
  ]
}
```

Status codes: `422` empty question · `503` vector store missing (run `ingest.py`) ·
`500` inference error. `GET /health` returns `{"status": "ok"}`.

---

## Configuration

Retrieval behaviour is tuned by the constants at the top of `rag.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `RETRIEVE_K` | `20` | Candidates pulled from ChromaDB before filtering. |
| `SCORE_THRESHOLD` | `0.10` | Minimum relevance score; below this a chunk is discarded. |
| `TOP_K` | `5` | Chunks sent to the LLM after reranking. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for reranking. |

Chunking is configured in `ingest.py` (`CHUNK_SIZE`, `CHUNK_OVERLAP`). Changing either
requires a re-ingest: `python start.py --reingest`.

The only required environment variable is `OPENAI_API_KEY`, read from `.env`.

---

## Evaluation

```bash
python evaluate.py
```

Runs [RAGAs](https://docs.ragas.io) over the 28 questions in `EVAL_QUESTIONS`, scoring
**faithfulness** (are the claims grounded in the retrieved context?) and **answer
relevancy**. Results are written to `evaluation_results.json`; if that file already exists,
the run prints a before/after table so you can see whether a retrieval change helped.

The questions span the income statement, balance sheet, cash flow statement, segment note,
and narrative sections, and their ground truths were read directly from Microsoft's FY2024
10-K. Point the pipeline at a different filing and you must replace the whole list.

Two things to know before reading the numbers:

- **A correct refusal scores 0.0 on answer relevancy.** RAGAs treats "not found in
  context" as non-committal and scores it zero, so the pipeline is penalised for exactly
  the behaviour the strict prompt is meant to produce.
- **The eval must see the same context the model saw.** `evaluate.py` builds contexts with
  `rag.doc_to_context()`, which prefixes each chunk with its source and page. Passing the
  bare `page_content` makes the page citation the prompt mandates look like an unsupported
  claim, which pins faithfulness near 0.5 for every correctly-cited answer.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover the pure chunking logic and need no API key or network access.

---

## Notes and limitations

- Costs money: ingestion embeds every chunk, and each query calls `gpt-4o-mini`.
- The vector store (`chroma_db/`) and source PDFs are gitignored — both are rebuilt
  locally from your own filings.
- The reranker downloads a ~90 MB model from Hugging Face on first use.
- Paths are relative, so run the scripts from the repository root.
- Answers are generated from a limited set of retrieved excerpts. This is a demonstration
  project, not financial advice — verify any figure against the filing before relying on it.

---

## License

[MIT](LICENSE)
