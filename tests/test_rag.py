"""Tests for context formatting in rag.py.

doc_to_context is shared between the LLM prompt and the RAGAs evaluation, so
these pin both the provenance line and the prompt layout. No API key needed:
nothing here touches the vector store or the model.
"""

from langchain_core.documents import Document

from rag import _format_docs, doc_to_context


def _doc(content="Total revenue 245,122", page=66, source="microsoft-sec-10k.pdf"):
    return Document(page_content=content, metadata={"source_file": source, "page": page})


def test_context_carries_source_and_page():
    out = doc_to_context(_doc())
    assert out == "Source: microsoft-sec-10k.pdf, Page: 66\nTotal revenue 245,122"


def test_context_falls_back_to_the_source_metadata_key():
    doc = Document(page_content="body", metadata={"source": "other.pdf", "page": 3})
    assert doc_to_context(doc).startswith("Source: other.pdf, Page: 3")


def test_context_tolerates_missing_metadata():
    assert doc_to_context(Document(page_content="body")) == "Source: unknown, Page: ?\nbody"


def test_format_docs_numbers_entries_and_keeps_the_separator():
    """Pins the exact prompt layout - changing it silently changes the prompt."""
    out = _format_docs([_doc("first", page=1), _doc("second", page=2)])
    assert out == (
        "[1] Source: microsoft-sec-10k.pdf, Page: 1\nfirst"
        "\n\n---\n\n"
        "[2] Source: microsoft-sec-10k.pdf, Page: 2\nsecond"
    )


def test_format_docs_uses_the_same_provenance_line_as_the_evaluation():
    """The eval passes doc_to_context(doc); the prompt must contain that verbatim."""
    doc = _doc()
    assert doc_to_context(doc) in _format_docs([doc])
