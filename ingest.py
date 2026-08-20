"""
ingest.py — Load 10-K PDFs from ./data, chunk them, embed with OpenAI,
and persist to ChromaDB.

Chunking strategy:
  - Token-aware splitting via tiktoken (cl100k_base), 1024 tokens / 128 overlap.
  - Table boundary protection: consecutive lines that look like financial table
    rows are temporarily joined so the splitter never cuts across them.

Usage:
    python ingest.py
"""

import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_reports"

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
                    # Blank line after the table block to signal a paragraph break
                    if out and not out[-1].endswith(_TABLE_NL):
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

def load_pdfs(data_dir: Path) -> list:
    docs = []
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir.resolve()}")
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        docs.extend(pages)
    print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDF(s).")
    return docs


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
    print(f"Split into {len(chunks)} chunks (1024 tok / 128 overlap, table-aware).")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embedding chunks and storing in ChromaDB …")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"Vectorstore persisted to {CHROMA_DIR.resolve()}")
    return vectorstore


def main():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"Created {DATA_DIR.resolve()} — place your 10-K PDF files there and re-run.")
        return

    docs = load_pdfs(DATA_DIR)
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
