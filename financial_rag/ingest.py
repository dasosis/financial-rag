"""
ingest.py — Load 10-K filings from ./data, chunk them, embed with OpenAI,
and persist to ChromaDB. Re-running is idempotent: each file's chunks
replace any previously ingested from a file of the same name and owner.

A filing is a PDF or the HTML 10-K EDGAR serves (.htm/.html; see
html_filing.py). Every chunk carries an `owner`:
  - data/*.pdf, *.htm        owner "public": sample filings every user can query
  - data/users/<id>/...      owner <id>: one user's private uploads and fetches

api.py reuses ingest_file() to add user-uploaded PDFs one at a time.

Each file's financial statements are also extracted into structured figures
(see financials.py) for the deterministic analysis engine.

Chunking strategy:
  - Token-aware splitting via tiktoken (cl100k_base), 1024 tokens / 128 overlap.
  - Table boundary protection: consecutive lines that look like financial table
    rows are temporarily joined so the splitter never cuts across them.
  - HTML filings: a chunk holding a table's rows but not its column headings
    gets the headings repeated at its top, so "57,582 50,901" says which year.

A public sample removed from data/ is removed from the store too, chunks and
figures, when ingest next runs.

The samples are Microsoft's FY2024 and Apple's FY2025 10-Ks as EDGAR serves them
(SAMPLES), the filings eval_answers.py's questions were checked against.

Usage:
    python -m financial_rag.ingest                    # (re)ingest everything in data/
    python -m financial_rag.ingest --fetch-samples    # download the samples first
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from financial_rag import config, financials, html_filing
from financial_rag.tracing import note, traced

load_dotenv()

DATA_DIR = Path("data")
USERS_DIR = DATA_DIR / "users"
PUBLIC_OWNER = "public"
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_reports"

FILING_SUFFIXES = (".pdf", ".htm", ".html")

# The public samples, saved under data/ by --fetch-samples
SAMPLES = {
    "microsoft-sec-10k.htm":
        "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
    "apple-sec-10k.htm":
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
}

CHUNK_SIZE = 1024    # tokens
CHUNK_OVERLAP = 128  # tokens

# Placeholder used to hide intra-table newlines from the splitter
_TABLE_NL = "\u2060NL\u2060"   # word-joiner + NL + word-joiner (invisible in embedding)


# ---------------------------------------------------------------------------
# Table boundary protection
# ---------------------------------------------------------------------------

def _is_table_line(line: str) -> bool:
    """
    Return True if the line looks like part of a financial table row:
      - Contains a $ sign plus at least one number, or
      - Contains 3+ comma-formatted numbers (e.g. 245,122), or
      - Is a row of dashes/spaces used as a separator between table rows.
    """
    s = line.strip()
    if not s:
        return False
    # Dash-separator rows  (e.g. "  --------   --------   --------")
    if re.fullmatch(r'[\s\-\$=]+', s) and len(s) > 4:
        return True
    # Count large financial numbers:  1,234  or  1,234,567
    big_numbers = re.findall(r'\b\d{1,3}(?:,\d{3})+\b', s)
    dollars = s.count("$")
    if dollars >= 1 and len(big_numbers) >= 1:
        return True
    if len(big_numbers) >= 3:
        return True
    return False


def _protect_table_blocks(docs: list) -> list:
    """
    Within each document's page_content, replace newlines that fall *inside*
    a contiguous block of table-like lines with _TABLE_NL so the splitter
    treats the whole block as one logical unit.
    """
    for doc in docs:
        lines = doc.page_content.split("\n")
        out: list[str] = []
        in_table = False
        for line in lines:
            if _is_table_line(line):
                if not in_table:
                    in_table = True
                    # Make sure we don't stick the table onto the previous para
                    if out and out[-1] != "":
                        out.append("")
                # Store the line with its trailing newline replaced by placeholder
                out.append(line + _TABLE_NL)
            else:
                if in_table:
                    in_table = False
                    # Blank line after the table block to signal a paragraph break.
                    # Skipped when the terminating line is itself blank, since it
                    # already provides the break.
                    if line != "":
                        out.append("")
                out.append(line)
        doc.page_content = "\n".join(out)
    return docs


def _restore_table_blocks(chunks: list) -> list:
    """Restore the placeholder newlines in each chunk after splitting."""
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace(_TABLE_NL, "\n")
    return chunks


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_html(path: Path, owner: str) -> tuple[list, dict[int, list[html_filing.Table]]]:
    """An EDGAR HTML filing as one Document per page, plus each page's tables."""
    print(f"Loading {path.name} ...")
    pages = html_filing.load_pages(path.read_text(encoding="utf-8", errors="replace"))
    docs = [
        Document(page_content=page.text,
                 metadata={"page": page.number, "source_file": path.name, "owner": owner})
        for page in pages
    ]
    return docs, {page.number: page.tables for page in pages}


def load_filing(path: Path, owner: str) -> tuple[list, dict[int, list[html_filing.Table]]]:
    """One Document per page, plus the tables on each page (HTML only)."""
    if path.suffix.lower() in (".htm", ".html"):
        return load_html(path, owner)
    return load_pdf(path, owner), {}


def load_pdf(pdf_path: Path, owner: str) -> list:
    print(f"Loading {pdf_path.name} ...")
    pages = PyPDFLoader(str(pdf_path)).load()
    for page in pages:
        page.metadata["source_file"] = pdf_path.name
        page.metadata["owner"] = owner
    return pages


def split_documents(docs: list) -> list:
    # Protect table rows before splitting
    docs = _protect_table_blocks(docs)

    # Token-aware splitter using the same tokeniser as text-embedding-3-small
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Restore placeholder newlines inside table blocks
    chunks = _restore_table_blocks(chunks)
    return chunks


def open_vectorstore() -> Chroma:
    """Open the persisted store, creating it on first write if it doesn't exist."""
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=config.embeddings(),
        collection_name=COLLECTION_NAME,
    )


@traced("ingest")
def ingest_file(
    pdf_path: Path, owner: str = PUBLIC_OWNER, max_pages: int | None = None,
    cik: int | None = None,
) -> int:
    """
    Chunk and embed one PDF into the store, returning the number of chunks added,
    then extract its financial statements into structured figures.

    A file longer than max_pages is rejected before anything is embedded: the
    cost of ingesting grows with its length. `cik`, when known (a filing fetched
    by ticker), names the SEC filer whose XBRL data checks the figures.

    Chunks and figures previously ingested from a file with the same name and
    owner are replaced, so re-ingesting (or re-uploading) a filing never
    duplicates passages or keeps stale numbers. Other owners' files of the same
    name are untouched.
    """
    docs, tables = load_filing(pdf_path, owner)
    if max_pages is not None and len(docs) > max_pages:
        raise ValueError(
            f"{pdf_path.name} has {len(docs)} pages; the limit is {max_pages}."
        )
    if not any(doc.page_content.strip() for doc in docs):
        raise ValueError(
            f"{pdf_path.name} has no extractable text (scanned PDFs need OCR first)."
        )
    # Copy the page text first: split_documents rewrites table newlines in place.
    pages = {doc.metadata["page"]: doc.page_content for doc in docs}
    chunks = split_documents(docs)
    for chunk in chunks:
        on_page = tables.get(chunk.metadata["page"])
        if on_page:
            chunk.page_content = html_filing.with_table_headings(chunk.page_content, on_page)
    company = registrant_name(pages.get(0, ""))
    if company:
        for chunk in chunks:
            chunk.metadata["company"] = company

    vectorstore = open_vectorstore()
    stale = vectorstore.get(
        where={"$and": [{"source_file": pdf_path.name}, {"owner": owner}]}, include=[]
    )["ids"]
    if stale:
        print(f"Replacing {len(stale)} existing chunks from {pdf_path.name}.")
        vectorstore.delete(ids=stale)
    print(f"Embedding {len(chunks)} chunks from {pdf_path.name} …")
    vectorstore.add_documents(chunks)

    # Figures are an extra: if extraction fails, text Q&A still works, and the
    # analysis engine reports the figures as missing rather than guessing.
    try:
        financials.extract_and_store(pages, pdf_path.name, owner, company, cik)
    except Exception as exc:
        print(f"Warning: financial figure extraction failed for {pdf_path.name}: {exc}")
    note(source_file=pdf_path.name, pages=len(docs), chunks=len(chunks))
    return len(chunks)


# ---------------------------------------------------------------------------
# Filings and their companies
# ---------------------------------------------------------------------------

_REGISTRANT_NOTE = re.compile(r"exact name of registrant", re.IGNORECASE)
_FILE_NUMBER = re.compile(r"commission file number", re.IGNORECASE)


def registrant_name(cover: str) -> str | None:
    """
    The company a 10-K belongs to, read from its cover page.

    Most covers print the name above "(Exact name of registrant as specified in
    its charter)"; some, like Microsoft's, print it straight after the
    "Commission File Number" line instead. All-capital names are title-cased.
    """
    lines = [" ".join(line.split()) for line in cover.split("\n")]
    lines = [line for line in lines if line]
    name = None
    for i, line in enumerate(lines):
        if _REGISTRANT_NOTE.search(line):
            before = _REGISTRANT_NOTE.split(line)[0].strip(" (")
            name = before or (lines[i - 1] if i else None)
            break
    else:
        for i, line in enumerate(lines[:-1]):
            if _FILE_NUMBER.search(line):
                name = lines[i + 1]
                break
    if not name or len(name) > 80 or not re.search(r"[A-Za-z]{2}", name):
        return None
    return name.title() if name == name.upper() else name


@dataclass(frozen=True)
class Filing:
    source_file: str
    owner: str
    chunks: int
    company: str | None


_companies: dict[tuple[str, str, int], str | None] = {}


def filings(owners: list[str]) -> list[Filing]:
    """
    Every filing the given owners have ingested, with its chunk count and company.

    The company comes from the chunks' "company" metadata (set at ingest), or for
    filings ingested before that, from the cover page text, cached per filing.
    """
    if not CHROMA_DIR.exists():
        return []
    store = open_vectorstore()
    got = store.get(where={"owner": {"$in": owners}}, include=["metadatas"])
    counts: dict[tuple[str, str], int] = {}
    companies: dict[tuple[str, str], str] = {}
    for meta in got["metadatas"]:
        key = (meta.get("owner", ""), meta.get("source_file", "unknown"))
        counts[key] = counts.get(key, 0) + 1
        if meta.get("company"):
            companies[key] = meta["company"]
    out = []
    for (owner, name), n in sorted(counts.items(), key=lambda kv: kv[0][1]):
        company = companies.get((owner, name))
        if company is None:
            cache_key = (owner, name, n)
            if cache_key not in _companies:
                cover = store.get(
                    where={"$and": [{"owner": owner}, {"source_file": name}, {"page": 0}]},
                    include=["documents"],
                )["documents"]
                _companies[cache_key] = registrant_name("\n".join(cover))
            company = _companies[cache_key]
        out.append(Filing(name, owner, n, company))
    return out


def list_documents(owner: str) -> dict[str, int]:
    """Map each source file the owner has ingested to its chunk count."""
    if not CHROMA_DIR.exists():
        return {}
    metadatas = open_vectorstore().get(where={"owner": owner}, include=["metadatas"])[
        "metadatas"
    ]
    counts: dict[str, int] = {}
    for meta in metadatas:
        name = meta.get("source_file", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _filings_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in FILING_SUFFIXES)


def filing_jobs() -> list[tuple[Path, str]]:
    """Every filing under data/ with its owner: public samples, then each user's."""
    jobs = [(p, PUBLIC_OWNER) for p in _filings_in(DATA_DIR)] if DATA_DIR.exists() else []
    if USERS_DIR.exists():
        for user_dir in sorted(p for p in USERS_DIR.iterdir() if p.is_dir()):
            jobs += [(p, user_dir.name) for p in _filings_in(user_dir)]
    return jobs


def fetch_samples() -> None:
    """Download any sample filing data/ doesn't have yet (needs SEC_USER_AGENT)."""
    from financial_rag import edgar  # local import: only this command talks to EDGAR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLES.items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"Downloading {name} from {url} …")
            path.write_text(edgar.fetch_document(url), encoding="utf-8")


def remove_file(name: str, owner: str) -> int:
    """Remove one ingested file's chunks and figures; returns the chunks removed."""
    vectorstore = open_vectorstore()
    ids = vectorstore.get(
        where={"$and": [{"source_file": name}, {"owner": owner}]}, include=[]
    )["ids"]
    if ids:
        vectorstore.delete(ids=ids)
    financials.replace_facts(owner, name, [])
    return len(ids)


def prune_samples() -> list[str]:
    """Remove public samples no longer in data/ from the store; returns their names."""
    on_disk = {p.name for p in _filings_in(DATA_DIR)}
    gone = [name for name in list_documents(PUBLIC_OWNER) if name not in on_disk]
    for name in gone:
        print(f"Removing {remove_file(name, PUBLIC_OWNER)} chunks of {name}: "
              f"no longer in {DATA_DIR}/.")
    return gone


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Ingest the filings in data/.")
    parser.add_argument("--fetch-samples", action="store_true",
                        help="download the sample 10-Ks from SEC EDGAR first")
    args = parser.parse_args(argv)
    config.run_as("system:ingest")
    if args.fetch_samples:
        fetch_samples()
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"Created {DATA_DIR.resolve()} — place your 10-K filings there and re-run.")
        return

    # Rebuilding the store from scratch restores everything with its access.
    jobs = filing_jobs()
    if not jobs:
        raise FileNotFoundError(f"No filings found in {DATA_DIR.resolve()}")
    prune_samples()
    for pdf_path, owner in jobs:
        ingest_file(pdf_path, owner)
    print(f"Vectorstore persisted to {CHROMA_DIR.resolve()}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
