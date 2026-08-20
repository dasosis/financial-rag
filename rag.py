"""
rag.py — Core RAG logic with score-threshold filtering and cross-encoder reranking.

Pipeline:
  1. Embed the question and retrieve top-RETRIEVE_K candidates from ChromaDB.
  2. Drop any chunk whose cosine-similarity relevance score < SCORE_THRESHOLD (0.40).
  3. Rerank survivors with cross-encoder/ms-marco-MiniLM-L-6-v2.
  4. Keep top-TOP_K (5) chunks and send to GPT-4o-mini.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_reports"
RETRIEVE_K = 20           # candidates pulled from ChromaDB before filtering
SCORE_THRESHOLD = 0.10    # minimum relevance score (Chroma L2-normalised scores top ~0.16)
TOP_K = 5                 # final chunks sent to the LLM after reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions using ONLY \
the retrieved 10-K excerpts provided below.

STRICT RULES — follow these exactly:
1. Only report figures that appear VERBATIM in the excerpts. Do NOT compute, infer, \
   derive, or combine values from multiple rows unless the document explicitly states \
   the combined figure.
2. If a figure is not explicitly stated in the excerpts, respond with \
   "not found in context" for that specific item. Never guess or estimate.
3. Do not add commentary, context, or assumptions beyond what is directly stated.
4. After each claim, cite the source document and page in parentheses, \
   e.g. (microsoft-sec-10k.pdf, p.66).

Context excerpts:
{context}
"""

# ---------------------------------------------------------------------------
# Lazy cross-encoder singleton (loaded once per process)
# ---------------------------------------------------------------------------
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker model {RERANKER_MODEL} …")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieved_docs: list = field(default_factory=list)   # used by evaluate.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def doc_to_context(doc) -> str:
    """Render one retrieved chunk the way the LLM sees it: provenance + content.

    evaluate.py uses this too, so RAGAs judges the answer against exactly the
    context the model was given. Passing bare page_content instead makes the page
    citation the system prompt mandates look like an unsupported claim, which
    caps faithfulness at roughly 0.5 for every correctly-cited answer.
    """
    meta = doc.metadata
    source = meta.get("source_file", meta.get("source", "unknown"))
    page = meta.get("page", "?")
    return f"Source: {source}, Page: {page}\n{doc.page_content}"


def _format_docs(docs) -> str:
    parts = [f"[{i}] {doc_to_context(doc)}" for i, doc in enumerate(docs, 1)]
    return "\n\n---\n\n".join(parts)


def _answer_with_docs(question: str, docs: list) -> str:
    """Format context from pre-fetched docs and call GPT-4o-mini."""
    context = _format_docs(docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = (
        {"context": RunnableLambda(lambda _: context), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _load_vectorstore() -> Chroma:
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_DIR.resolve()}. "
            "Run `python ingest.py` first."
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def load_retriever():
    """Backward-compatible retriever for evaluate.py (bypasses reranker)."""
    return _load_vectorstore().as_retriever(search_kwargs={"k": TOP_K})


def retrieve_and_rerank(question: str, vectorstore: Chroma) -> list:
    """
    1. Fetch top-RETRIEVE_K chunks with relevance scores.
    2. Drop chunks below SCORE_THRESHOLD.
    3. Cross-encoder rerank, return top-TOP_K.
    """
    # Step 1 — semantic retrieval with relevance scores (0 = dissimilar, 1 = identical)
    scored = vectorstore.similarity_search_with_relevance_scores(
        question, k=RETRIEVE_K
    )

    # Step 2 — threshold filter
    filtered = [(doc, score) for doc, score in scored if score >= SCORE_THRESHOLD]
    if not filtered:
        return []

    # Step 3 — cross-encoder reranking
    reranker = _get_reranker()
    docs = [doc for doc, _ in filtered]
    ce_scores = reranker.predict([(question, doc.page_content) for doc in docs])

    ranked = sorted(zip(ce_scores, docs, strict=True), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:TOP_K]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query(question: str) -> RAGResponse:
    vectorstore = _load_vectorstore()
    top_docs = retrieve_and_rerank(question, vectorstore)

    if not top_docs:
        return RAGResponse(
            answer=(
                "No relevant context found above the confidence threshold. "
                "The documents may not contain information about this question."
            ),
            sources=[],
            retrieved_docs=[],
        )

    answer = _answer_with_docs(question, top_docs)

    sources = []
    seen: set = set()
    for doc in top_docs:
        meta = doc.metadata
        source = meta.get("source_file", meta.get("source", "unknown"))
        page = meta.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source_file": source,
                    "page": page,
                    "excerpt": doc.page_content[:300],
                }
            )

    return RAGResponse(answer=answer, sources=sources, retrieved_docs=top_docs)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What was the total revenue?"
    result = query(q)
    print(f"\nAnswer:\n{result.answer}\n")
    print("Sources:")
    for s in result.sources:
        print(f"  - {s['source_file']} p.{s['page']}: {s['excerpt'][:120]} …")
