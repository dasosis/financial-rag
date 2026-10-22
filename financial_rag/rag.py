"""
rag.py — Core RAG logic: hybrid (semantic + BM25) retrieval and cross-encoder reranking.

Pipeline:
  1. Semantic: embed the question, take the top RETRIEVE_K chunks from ChromaDB,
     and drop any whose relevance score is below SCORE_THRESHOLD.
  2. Keyword: BM25 over the same chunks (the caller's own filings plus the public
     samples) gives its top KEYWORD_K. It catches exact fiscal years, line-item
     labels and values that embeddings blur together.
  3. Fuse the two rankings (weighted reciprocal rank fusion; the keyword weight
     depends on the question, see hybrid.profile_query) and keep RERANK_CANDIDATES.
  4. Rerank those with cross-encoder/ms-marco-MiniLM-L-6-v2, then have
     gpt-4o-mini order the cross-encoder's top LLM_RERANK_CANDIDATES (15): it reads
     statements, line items and fiscal years far better than the small
     cross-encoder (see eval_retrieval.py). If that call fails, the cross-encoder
     order stands. When the question quotes exact values ("57,582"), chunks
     containing all of them go first either way.
  5. Send the top TOP_K (5) to GPT-4o-mini. Every returned chunk records how it
     was found under metadata["retrieval"], and the API passes that on.

HYBRID_SEARCH=0 turns step 2 off, leaving the semantic-only pipeline.
RERANKER=cross-encoder skips the LLM step in 4 (no extra model call per question).
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from financial_rag import config, hybrid, tracing
from financial_rag.financials import page_number
from financial_rag.ingest import PUBLIC_OWNER

load_dotenv()

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_reports"
RETRIEVE_K = 20           # semantic candidates pulled from ChromaDB before filtering
SCORE_THRESHOLD = 0.10    # minimum relevance score (Chroma L2-normalised scores top ~0.16)
KEYWORD_K = 20            # BM25 candidates
RERANK_CANDIDATES = 30    # fused candidates the cross-encoder scores
LLM_RERANK_CANDIDATES = 15  # of those, how many the LLM reranker orders
RERANKER = os.getenv("RERANKER", "llm")   # "llm" or "cross-encoder"
LLM_RERANK_TIMEOUT = 30   # seconds; on timeout the cross-encoder order is used
TOP_K = 5                 # final chunks sent to the LLM after reranking
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "1") != "0"

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions using ONLY \
the retrieved 10-K excerpts provided below.

STRICT RULES — follow these exactly:
1. Only report figures that appear VERBATIM in the excerpts. Do NOT compute, infer, \
   derive, or combine values from multiple rows unless the document explicitly states \
   the combined figure.
2. If a figure is not explicitly stated in the excerpts, respond with \
   "not found in context" for that specific item. Never guess or estimate.
3. For a question about what a filing says (its risks, strategy, policies, \
   business), summarise what the excerpts state, point by point. Answer \
   "not found in context" only when no excerpt addresses the question at all. \
   Do not add commentary, context, or assumptions beyond what the excerpts state.
4. After each claim, cite the source document and page in parentheses, \
   e.g. (microsoft-sec-10k.htm, p.67).
5. The excerpts may come from more than one filing, each named in its Source line. \
   If more than one filing answers the question, answer for each one separately \
   under its company or file name; never merge or combine figures across filings. \
   When the question asks how they compare or differ, end with the differences \
   the excerpts themselves show, each point cited to the filing it comes from.

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
        print(f"Loading reranker model {config.RERANKER_MODEL} …")
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


# ---------------------------------------------------------------------------
# LLM reranker
# ---------------------------------------------------------------------------

LLM_RERANK_PROMPT = """You are ranking excerpts from SEC 10-K filings by how well they \
answer a question.

Question: {question}

Rank the passages so the one that most directly answers the question comes first:
- Best: the passage states the exact figure or fact asked for: the right line item, the \
right fiscal year or date, and the right company if one is named.
- Next: passages with closely related figures, or the explanation behind them.
- When several passages contain the figure, prefer the primary financial statement or \
the note that reports it.
List every passage number that could help, best first. Leave out irrelevant passages.

Passages:

{passages}"""


class _PassageRanking(BaseModel):
    ranking: list[int] = Field(description="Passage numbers, most useful first")


def llm_rank(question: str, docs: list) -> list[int]:
    """
    Positions in `docs`, most useful first, as gpt-4o-mini orders them. Passages
    it leaves out are missing; numbers out of range or repeated are dropped.
    """
    passages = "\n\n".join(f"[{n}] {doc_to_context(doc)}" for n, doc in enumerate(docs, 1))
    llm = config.chat_model(
        temperature=0, timeout=LLM_RERANK_TIMEOUT, max_retries=1
    ).with_structured_output(_PassageRanking)
    result = llm.invoke(LLM_RERANK_PROMPT.format(question=question, passages=passages))
    order: list[int] = []
    for n in result.ranking:
        if 1 <= n <= len(docs) and n - 1 not in order:
            order.append(n - 1)
    return order


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
    if meta.get("company"):
        source = f"{source} ({meta['company']})"
    page = page_number(meta.get("page"))
    return f"Source: {source}, Page: {page}\n{strip_page_footers(doc.page_content)}"


# A filing's own page footer, "Apple Inc. | 2025 Form 10-K | 22": its number is the
# printed page, not the PDF page citations use, and the model would cite it.
_PAGE_FOOTER = re.compile(r"^[^\n]*\|\s*(?:\d{4}\s+)?Form\s+10-K\s*\|\s*\d+[ \t]*$",
                          re.MULTILINE | re.IGNORECASE)


def strip_page_footers(text: str) -> str:
    return _PAGE_FOOTER.sub("", text)


def _format_docs(docs) -> str:
    parts = [f"[{i}] {doc_to_context(doc)}" for i, doc in enumerate(docs, 1)]
    return "\n\n---\n\n".join(parts)


RETRY_PROMPT = """Your answer to that question failed a check against the excerpts:
{problems}
Answer the question again. Use only figures exactly as they are printed in the \
excerpts, and cite only a file and page shown on an excerpt's Source line."""


def _answer_with_docs(question: str, docs: list, problems: list[str] | None = None) -> str:
    """Format context from pre-fetched docs and call GPT-4o-mini.

    `problems` (from check_answer) asks for a corrected answer instead.
    """
    context = _format_docs(docs)
    messages = [("system", SYSTEM_PROMPT), ("human", "{question}")]
    if problems:
        messages.append(("human", RETRY_PROMPT))
    prompt = ChatPromptTemplate.from_messages(messages)
    llm = config.chat_model(temperature=0)
    chain = (
        {
            "context": RunnableLambda(lambda _: context),
            "question": RunnablePassthrough(),
            "problems": RunnableLambda(lambda _: "\n".join(f"- {p}" for p in problems or [])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    if not tracing.listened():
        return chain.invoke(question)
    # A streaming answer (api.py /query/stream) shows the text as it's written
    parts = []
    for part in chain.stream(question):
        parts.append(part)
        tracing.emit("token", part)
    return "".join(parts)


# Figures an answer must have taken from an excerpt: comma-grouped numbers and
# decimals ("50,969", "11.80", "$10.3 billion"). Bare integers (years, "16%")
# aren't checked.
_ANSWER_FIGURE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+")
# "(msft.pdf, p.90)", or an EDGAR HTML filing's "(microsoft-sec-10k.htm, p. 58)"
_CITATION = re.compile(r"\(([^()]+?\.(?:pdf|html?)),\s*p\.?\s*(\d+)\)", re.IGNORECASE)


def _numbers(text: str) -> set[str]:
    """Number tokens as hybrid.tokenize writes them: "$ 50,969" -> "50969"."""
    return {t for t in hybrid.tokenize(text) if t[0].isdigit()}


def check_answer(answer: str, question: str, docs: list) -> list[str]:
    """
    What in a text answer isn't backed by the excerpts it was written from.
    Empty when everything checks out.

    - Each "(file.pdf, p.N)" or "(file.htm, p.N)" must name a retrieved excerpt, and every figure in
      the text before it (back to the previous citation) must be printed on that
      page or in the question.
    - Figures with no citation after them must be printed in some excerpt.

    The prompt already forbids invented figures and citations; this catches the
    model doing it anyway, e.g. blending "Goodwill 50,969" and "$51.0 billion"
    into "$51,969 million", or citing the wrong page for a real figure.
    """
    asked = _numbers(question)

    # A page is cited by the number on its Source line (page_number), or by its
    # PDF page label where that differs, e.g. a roman-numbered front section
    on_page: dict[tuple[str, str], set[str]] = {}
    for doc in docs:
        source = doc.metadata.get("source_file", doc.metadata.get("source"))
        numbers = {page_number(doc.metadata.get("page")), doc.metadata.get("page_label")}
        for number in numbers - {None, "?"}:
            on_page.setdefault((source, str(number)), set()).update(_numbers(doc.page_content))
    anywhere = set().union(*on_page.values()) if on_page else set()

    problems: list[str] = []

    def check(text: str, printed: set[str], where: str) -> None:
        for figure in dict.fromkeys(_ANSWER_FIGURE.findall(text)):
            number = figure.replace(",", "")
            if number not in printed and number not in asked:
                problem = f"{figure} isn't printed {where}"
                if problem not in problems:
                    problems.append(problem)

    start = 0
    for match in _CITATION.finditer(answer):
        source, page = match.group(1).strip(), match.group(2)
        cited = on_page.get((source, page))
        if cited is None:
            problem = f"({source}, p.{page}) isn't the source of any excerpt"
            if problem not in problems:
                problems.append(problem)
            check(answer[start:match.start()], anywhere, "in any excerpt")
        else:
            check(answer[start:match.start()], cited, f"on {source} p.{page}")
        start = match.end()
    check(answer[start:], anywhere, "in any excerpt")
    return problems


def answer_checked(question: str, docs: list) -> str:
    """
    Answer, check the answer against the excerpts, and retry once with the
    problems spelled out. If the retry fails the check too, the answer goes out
    with a warning naming what couldn't be verified, rather than as if it were.
    """
    with tracing.tracer.start_as_current_span("generate"):
        tracing.note(attempt=1, excerpts=len(docs))
        answer = _answer_with_docs(question, docs)
    with tracing.tracer.start_as_current_span("check"):
        problems = check_answer(answer, question, docs)
        tracing.note(problems=len(problems))
    if not problems:
        return answer
    print(f"Answer failed the excerpt check, retrying: {problems}")
    tracing.emit("retry")   # a streamed first attempt is withdrawn
    with tracing.tracer.start_as_current_span("generate"):
        tracing.note(attempt=2, excerpts=len(docs))
        answer = _answer_with_docs(question, docs, problems)
    with tracing.tracer.start_as_current_span("check"):
        problems = check_answer(answer, question, docs)
        tracing.note(problems=len(problems), unverified=bool(problems))
    if problems:
        answer += (
            "\n\n⚠️ Not verified against the excerpts: " + "; ".join(problems)
            + ". Check the cited pages before relying on these figures."
        )
    return answer


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _load_vectorstore() -> Chroma:
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_DIR.resolve()}. "
            "Run `python -m financial_rag.ingest` first."
        )
    embeddings = config.embeddings()
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def load_retriever():
    """Backward-compatible retriever for evaluate.py (bypasses reranker)."""
    return _load_vectorstore().as_retriever(search_kwargs={"k": TOP_K})


def owner_filter(owner: str | None) -> dict | None:
    """Restrict retrieval to the owner's own filings plus the public samples.

    None searches every filing; only the CLI and evaluate.py do that.
    """
    if owner is None:
        return None
    return {"owner": {"$in": [owner, PUBLIC_OWNER]}}


# Keyword corpus: chunk id -> (chunk, BM25 term counts), per Chroma collection.
# Filled lazily from the store and kept in step with it, so a query only
# re-reads chunks added since the last one.
_corpora: dict[str, dict[str, tuple[Document, hybrid.Indexed]]] = {}
_corpora_lock = threading.Lock()
_FETCH_BATCH = 500


def _chunk_id(doc) -> str:
    """Chroma's id for the chunk, or a stand-in built from its provenance."""
    if doc.id:
        return doc.id
    meta = doc.metadata
    return f"{meta.get('owner')}/{meta.get('source_file')}/{meta.get('page')}/{hash(doc.page_content)}"


def _visible(meta: dict, owner: str | None) -> bool:
    return owner is None or meta.get("owner") in (owner, PUBLIC_OWNER)


@dataclass(frozen=True)
class Scope:
    """
    The chunks one search may see: the owner's filings plus the public samples,
    narrowed to a single filing when `filing` is set. A filing is (its owner,
    its source_file), and must itself be visible to the owner.
    """
    owner: str | None
    filing: tuple[str, str] | None = None

    def __post_init__(self):
        if self.filing and not _visible({"owner": self.filing[0]}, self.owner):
            raise ValueError("That filing belongs to another user.")

    def where(self) -> dict | None:
        if self.filing is None:
            return owner_filter(self.owner)
        file_owner, name = self.filing
        return {"$and": [{"owner": file_owner}, {"source_file": name}]}

    def contains(self, meta: dict) -> bool:
        if self.filing is not None:
            return (meta.get("owner"), meta.get("source_file")) == self.filing
        return _visible(meta, self.owner)


def _keyword_corpus(vectorstore: Chroma, scope: Scope) -> list[tuple[Document, hybrid.Indexed]]:
    """
    Every chunk in scope, with its BM25 term counts.

    Scope is applied by Chroma with the same filter as the semantic search, so
    the keyword path can't surface another user's chunks either.
    """
    ids = vectorstore.get(where=scope.where(), include=[])["ids"]
    with _corpora_lock:
        cache = _corpora.setdefault(vectorstore._collection.id, {})
        missing = [i for i in ids if i not in cache]
        for start in range(0, len(missing), _FETCH_BATCH):
            got = vectorstore.get(
                ids=missing[start:start + _FETCH_BATCH], include=["documents", "metadatas"]
            )
            for cid, text, meta in zip(got["ids"], got["documents"], got["metadatas"], strict=True):
                text = text or ""
                cache[cid] = (
                    Document(id=cid, page_content=text, metadata=meta or {}),
                    hybrid.index_text(text),
                )
        # Forget chunks in this scope that the store no longer has (replaced filings)
        present = set(ids)
        for cid in [c for c, (doc, _) in cache.items()
                    if c not in present and scope.contains(doc.metadata)]:
            del cache[cid]
        return [cache[i] for i in ids if i in cache]


def _keyword_search(profile: hybrid.QueryProfile, vectorstore: Chroma, scope: Scope):
    """Top KEYWORD_K chunks by BM25 as (chunk, score, matched terms); no-match chunks dropped."""
    corpus = _keyword_corpus(vectorstore, scope)
    scores = hybrid.bm25_scores(profile.terms, [indexed for _, indexed in corpus])
    ranked = sorted(
        ((score, doc, indexed) for score, (doc, indexed) in zip(scores, corpus, strict=True)
         if score > 0),
        key=lambda x: x[0],
        reverse=True,
    )[:KEYWORD_K]
    return [
        (doc, score, [t for t in profile.terms if t in indexed.tf])
        for score, doc, indexed in ranked
    ]


@tracing.traced("retrieve")
def retrieve_and_rerank(
    question: str,
    vectorstore: Chroma,
    owner: str | None = None,
    keyword: bool | None = None,
    filing: tuple[str, str] | None = None,
    reranker: str | None = None,
    top_k: int = TOP_K,
) -> list:
    """
    Hybrid retrieval for the chunks the owner may see; see the module docstring.

    Returns up to `top_k` chunks, best first. Each is a copy of the stored chunk
    whose metadata["retrieval"] records how it was found: chunk_id, matched_by
    ("semantic" and/or "keyword"), each search's rank and score, the query terms
    it matched, the cross-encoder score, and which reranker set the final order
    (with the LLM's rank when it did). `keyword` overrides HYBRID_SEARCH and
    `reranker` overrides RERANKER. `filing` = (owner, source_file) searches that
    one filing only.
    """
    reranker = RERANKER if reranker is None else reranker
    keyword = HYBRID_SEARCH if keyword is None else keyword
    profile = hybrid.profile_query(question)
    scope = Scope(owner, filing)

    # Semantic retrieval with relevance scores (0 = dissimilar, 1 = identical)
    with tracing.tracer.start_as_current_span("search.semantic"):
        scored = vectorstore.similarity_search_with_relevance_scores(
            question, k=RETRIEVE_K, filter=scope.where()
        )
        semantic = [(doc, score) for doc, score in scored if score >= SCORE_THRESHOLD]
        tracing.note(candidates=len(scored), above_threshold=len(semantic))
    if keyword:
        with tracing.tracer.start_as_current_span("search.keyword"):
            lexical = _keyword_search(profile, vectorstore, scope)
            tracing.note(hits=len(lexical))
    else:
        lexical = []

    docs: dict[str, Document] = {}
    trace: dict[str, dict] = {}

    def note(doc, via: str, **fields) -> str:
        cid = _chunk_id(doc)
        docs.setdefault(cid, doc)
        entry = trace.setdefault(cid, {"chunk_id": cid, "matched_by": []})
        entry["matched_by"].append(via)
        entry.update(fields)
        return cid

    semantic_ids = [
        note(doc, "semantic", semantic_rank=rank, semantic_score=round(score, 4))
        for rank, (doc, score) in enumerate(semantic, 1)
    ]
    keyword_ids = [
        note(doc, "keyword", keyword_rank=rank, keyword_score=round(score, 4),
             matched_terms=terms)
        for rank, (doc, score, terms) in enumerate(lexical, 1)
    ]
    if not docs:
        return []

    fused = hybrid.rrf(
        [(semantic_ids, hybrid.SEMANTIC_WEIGHT), (keyword_ids, profile.keyword_weight)]
    )[:RERANK_CANDIDATES]

    # Cross-encoder reranking of the fused candidates; a chunk containing every
    # value the question quotes outranks one that doesn't, whatever the model says
    cross_encoder = _get_reranker()
    candidates = [cid for cid, _ in fused]
    with tracing.tracer.start_as_current_span("rerank.cross_encoder"):
        ce_scores = cross_encoder.predict(
            [(question, docs[cid].page_content) for cid in candidates])
        tracing.note(candidates=len(candidates))
    exact = {
        cid: keyword and hybrid.has_amounts(profile, hybrid.index_text(docs[cid].page_content))
        for cid in candidates
    }
    ce = {cid: float(score) for cid, score in zip(candidates, ce_scores, strict=True)}
    ranked = sorted(candidates, key=lambda cid: (exact[cid], ce[cid]), reverse=True)

    # The LLM orders the cross-encoder's best few; the rest keep their order after
    # them. Exact-value matches still go first.
    used, llm_ranks = "cross-encoder", {}
    if reranker == "llm" and len(ranked) > 1:
        head = ranked[:LLM_RERANK_CANDIDATES]
        try:
            with tracing.tracer.start_as_current_span("rerank.llm"):
                tracing.note(candidates=len(head))
                order = [head[i] for i in llm_rank(question, [docs[cid] for cid in head])]
        except config.GatewayLimit:
            raise   # a spent budget is the user's answer, not a reason to degrade it
        except Exception as exc:
            print(f"LLM reranking failed, using the cross-encoder order: {exc}")
        else:
            llm_ranks = {cid: n for n, cid in enumerate(order, 1)}
            ranked = order + [cid for cid in ranked if cid not in llm_ranks]
            ranked = sorted(ranked, key=lambda cid: not exact[cid])   # stable
            used = "llm"

    out = []
    for cid in ranked[:top_k]:
        doc = docs[cid]
        info = {**trace[cid], "rerank_score": round(ce[cid], 4), "reranker": used}
        if cid in llm_ranks:
            info["llm_rank"] = llm_ranks[cid]
        if exact[cid]:
            info["exact_value_match"] = True
        out.append(Document(id=doc.id, page_content=doc.page_content,
                            metadata={**doc.metadata, "retrieval": info}))
    tracing.note(reranker=used, returned=len(out), scoped_to_filing=filing is not None)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def per_filing_k(count: int) -> int:
    """Chunks retrieved from each of `count` filings compared: enough for each to
    be represented, without crowding the prompt (3 each for two, 2 each for more)."""
    return max(2, -(-(TOP_K + 1) // count))


@tracing.traced("text_answer")
def query(
    question: str, owner: str | None = None, filing: tuple[str, str] | None = None,
    filings: list[tuple[str, str]] | None = None,
) -> RAGResponse:
    """
    Answer from the owner's filings and the samples, or from one `filing` only.

    `filings` (a comparison) searches each of several filings separately, so a
    question about two companies gets context from both rather than whichever
    one ranks higher.
    """
    vectorstore = _load_vectorstore()
    if filings:
        k = per_filing_k(len(filings))
        top_docs = [doc for one in filings
                    for doc in retrieve_and_rerank(question, vectorstore, owner, filing=one,
                                                   top_k=k)]
    else:
        top_docs = retrieve_and_rerank(question, vectorstore, owner, filing=filing)
    tracing.note(filings=len(filings) if filings else int(filing is not None), excerpts=len(top_docs))

    if not top_docs:
        return RAGResponse(
            answer=(
                "No relevant context found above the confidence threshold. "
                "The documents may not contain information about this question."
            ),
            sources=[],
            retrieved_docs=[],
        )

    answer = answer_checked(question, top_docs)
    return RAGResponse(
        answer=answer, sources=build_sources(question, top_docs), retrieved_docs=top_docs
    )


def build_sources(question: str, docs: list) -> list[dict]:
    """
    One citation per (file, page), in rank order, with how its chunk was found.

    The excerpt is the stretch of the chunk that matches the question, so a
    figure that was found by keyword is visible in the citation, not cut off.
    """
    terms = hybrid.profile_query(question).terms
    sources = []
    seen: set = set()
    for doc in docs:
        meta = doc.metadata
        source = meta.get("source_file", meta.get("source", "unknown"))
        page = page_number(meta.get("page"))
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source_file": source,
                    "page": page,
                    "excerpt": hybrid.focus(doc.page_content, terms)[:300],
                    "retrieval": meta.get("retrieval"),
                }
            )
    return sources

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What was the total revenue?"
    result = query(q)
    print(f"\nAnswer:\n{result.answer}\n")
    print("Sources:")
    for s in result.sources:
        found = s["retrieval"] or {}
        how = " + ".join(found.get("matched_by", []))
        print(f"  - {s['source_file']} p.{s['page']} [{how}]: {s['excerpt'][:120]} …")
