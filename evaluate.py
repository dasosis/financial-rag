"""
evaluate.py — RAGAs evaluation of the Financial Report Q&A Engine.

Metrics evaluated:
  - faithfulness       (are claims grounded in retrieved context?)
  - answer_relevancy   (does the answer address the question?)

Prints a before/after comparison table when a previous evaluation_results.json exists.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

import rag

load_dotenv()

EVAL_QUESTIONS: list[dict] = [
    {
        "question": "What was the total revenue for the most recent fiscal year?",
        "ground_truth": "Replace this with the actual ground truth from the 10-K.",
    },
    {
        "question": "What are the main risk factors disclosed in the filing?",
        "ground_truth": "Replace this with the actual ground truth from the 10-K.",
    },
    {
        "question": "What is the company's net income?",
        "ground_truth": "Replace this with the actual ground truth from the 10-K.",
    },
    {
        "question": "Describe the company's business segments.",
        "ground_truth": "Replace this with the actual ground truth from the 10-K.",
    },
    {
        "question": "What are the capital expenditures reported?",
        "ground_truth": "Replace this with the actual ground truth from the 10-K.",
    },
]

RESULTS_PATH = Path("evaluation_results.json")


def build_eval_dataset(questions: list[dict]) -> Dataset:
    rows = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    for item in questions:
        q = item["question"]
        print(f"  Querying: {q[:80]} …")
        # rag.query() now uses threshold filtering + reranker and exposes retrieved_docs
        result = rag.query(q)
        contexts = [doc.page_content for doc in result.retrieved_docs]

        rows["user_input"].append(q)
        rows["response"].append(result.answer)
        rows["retrieved_contexts"].append(contexts)
        rows["reference"].append(item["ground_truth"])

    return Dataset.from_dict(rows)


def _comparison_table(before: dict, after_df) -> None:
    """Print a side-by-side before/after table."""
    before_rows = {r["user_input"]: r for r in before.get("per_question", [])}

    col_w = 54
    print("\n" + "=" * 110)
    print(f"{'Question':<{col_w}}  {'Faith BEFORE':>12}  {'Faith AFTER':>11}  {'Rel BEFORE':>10}  {'Rel AFTER':>9}")
    print("=" * 110)

    for _, row in after_df.iterrows():
        q = row["user_input"]
        q_short = q[:col_w - 1] if len(q) > col_w else q
        b = before_rows.get(q, {})
        b_faith = b.get("faithfulness", float("nan"))
        b_rel = b.get("answer_relevancy", float("nan"))
        a_faith = row["faithfulness"]
        a_rel = row["answer_relevancy"]
        delta_f = a_faith - b_faith
        delta_r = a_rel - b_rel
        faith_arrow = "+" if delta_f > 0.01 else ("-" if delta_f < -0.01 else "=")
        rel_arrow = "+" if delta_r > 0.01 else ("-" if delta_r < -0.01 else "=")
        print(
            f"{q_short:<{col_w}}  {b_faith:>12.3f}  {a_faith:>9.3f}{faith_arrow}"
            f"  {b_rel:>10.3f}  {a_rel:>7.3f}{rel_arrow}"
        )

    before_agg = before.get("aggregate", {})
    b_f_mean = before_agg.get("faithfulness_mean", float("nan"))
    b_r_mean = before_agg.get("answer_relevancy_mean", float("nan"))
    a_f_mean = float(after_df["faithfulness"].mean())
    a_r_mean = float(after_df["answer_relevancy"].mean())

    print("-" * 110)
    faith_arrow = "+" if a_f_mean - b_f_mean > 0.005 else ("-" if a_f_mean - b_f_mean < -0.005 else "=")
    rel_arrow = "+" if a_r_mean - b_r_mean > 0.005 else ("-" if a_r_mean - b_r_mean < -0.005 else "=")
    print(
        f"{'MEAN':<{col_w}}  {b_f_mean:>12.3f}  {a_f_mean:>9.3f}{faith_arrow}"
        f"  {b_r_mean:>10.3f}  {a_r_mean:>7.3f}{rel_arrow}"
    )
    print("=" * 110)


def main():
    # Snapshot existing results as "before" baseline
    before_data = None
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            before_data = json.load(f)
        print(f"Loaded before-scores from {RESULTS_PATH}")

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    print("\nBuilding evaluation dataset …")
    dataset = build_eval_dataset(EVAL_QUESTIONS)

    print("\nRunning RAGAs evaluation (faithfulness + answer_relevancy) …")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    scores_df = results.to_pandas()

    print("\n=== After Scores ===")
    print(scores_df[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))

    agg = {
        "faithfulness_mean": float(scores_df["faithfulness"].mean()),
        "answer_relevancy_mean": float(scores_df["answer_relevancy"].mean()),
    }
    print(f"\nAggregate: {agg}")

    output = {
        "aggregate": agg,
        "per_question": scores_df.to_dict(orient="records"),
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"Full results saved to {RESULTS_PATH.resolve()}")

    if before_data:
        _comparison_table(before_data, scores_df)


if __name__ == "__main__":
    main()
