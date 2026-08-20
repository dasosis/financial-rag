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

# Ground truths below were read directly from data/microsoft-sec-10k.pdf (Microsoft
# FY2024 Form 10-K). Swap the whole list when you point the pipeline at another filing.
# Amounts are in millions of USD unless stated otherwise.
EVAL_QUESTIONS: list[dict] = [
    # --- Income statement -------------------------------------------------
    {
        "question": "What was total revenue for fiscal year 2024?",
        "ground_truth": "Total revenue was $245,122 million for the year ended June 30, 2024.",
    },
    {
        "question": "What was net income for fiscal year 2024?",
        "ground_truth": "Net income was $88,136 million for the year ended June 30, 2024.",
    },
    {
        "question": "What was operating income for fiscal year 2024?",
        "ground_truth": "Operating income was $109,433 million for the year ended June 30, 2024.",
    },
    {
        "question": "What was gross margin for fiscal year 2024?",
        "ground_truth": "Gross margin was $171,008 million for the year ended June 30, 2024.",
    },
    {
        "question": "What was total cost of revenue for fiscal year 2024?",
        "ground_truth": "Total cost of revenue was $74,114 million for the year ended June 30, 2024.",
    },
    {
        "question": "How much did the company spend on research and development in fiscal year 2024?",
        "ground_truth": "Research and development expense was $29,510 million in fiscal year 2024.",
    },
    {
        "question": "What were sales and marketing expenses in fiscal year 2024?",
        "ground_truth": "Sales and marketing expense was $24,456 million in fiscal year 2024.",
    },
    {
        "question": "What was the provision for income taxes in fiscal year 2024?",
        "ground_truth": "The provision for income taxes was $19,651 million in fiscal year 2024.",
    },
    {
        "question": "What was diluted earnings per share for fiscal year 2024?",
        "ground_truth": "Diluted earnings per share was $11.80 (basic was $11.86) for fiscal year 2024.",
    },

    # --- Balance sheet ----------------------------------------------------
    {
        "question": "What were total assets as of June 30, 2024?",
        "ground_truth": "Total assets were $512,163 million as of June 30, 2024.",
    },
    {
        "question": "How much cash and cash equivalents were held as of June 30, 2024?",
        "ground_truth": "Cash and cash equivalents were $18,315 million as of June 30, 2024.",
    },
    {
        "question": "What was the goodwill balance as of June 30, 2024?",
        "ground_truth": "Goodwill was $119,220 million as of June 30, 2024.",
    },
    {
        "question": "What was property and equipment, net of accumulated depreciation, as of June 30, 2024?",
        "ground_truth": "Property and equipment, net of accumulated depreciation of $76,421 million, was $135,591 million as of June 30, 2024.",
    },
    {
        "question": "What was the short-term unearned revenue balance as of June 30, 2024?",
        "ground_truth": "Short-term unearned revenue was $57,582 million as of June 30, 2024.",
    },

    # --- Cash flows -------------------------------------------------------
    {
        "question": "How much net cash was generated from operations in fiscal year 2024?",
        "ground_truth": "Net cash from operations was $118,548 million in fiscal year 2024.",
    },
    {
        "question": "What were additions to property and equipment in fiscal year 2024?",
        "ground_truth": "Additions to property and equipment were $44,477 million in fiscal year 2024.",
    },
    {
        "question": "How much common stock was repurchased in fiscal year 2024?",
        "ground_truth": "Common stock repurchased was $17,254 million in fiscal year 2024.",
    },
    {
        "question": "How much was paid in common stock cash dividends in fiscal year 2024?",
        "ground_truth": "Common stock cash dividends paid were $21,771 million in fiscal year 2024.",
    },
    {
        "question": "What was stock-based compensation expense in fiscal year 2024?",
        "ground_truth": "Stock-based compensation expense was $10,734 million in fiscal year 2024.",
    },

    # --- Segments and product revenue -------------------------------------
    {
        "question": "What are the company's reportable segments?",
        "ground_truth": "The three reportable segments are Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
    },
    {
        "question": "What was Intelligent Cloud segment revenue in fiscal year 2024?",
        "ground_truth": "Intelligent Cloud revenue was $105,362 million in fiscal year 2024.",
    },
    {
        "question": "What was operating income for the Productivity and Business Processes segment in fiscal year 2024?",
        "ground_truth": "Productivity and Business Processes operating income was $40,540 million in fiscal year 2024.",
    },
    {
        "question": "What was revenue from server products and cloud services in fiscal year 2024?",
        "ground_truth": "Server products and cloud services revenue was $97,726 million in fiscal year 2024.",
    },
    {
        "question": "What was Microsoft Cloud revenue in fiscal year 2024?",
        "ground_truth": "Microsoft Cloud revenue was $137.4 billion in fiscal year 2024.",
    },
    {
        "question": "How much revenue came from the United States in fiscal year 2024?",
        "ground_truth": "United States revenue was $124,704 million in fiscal year 2024.",
    },

    # --- Narrative sections -----------------------------------------------
    {
        "question": "How many people did the company employ as of June 30, 2024?",
        "ground_truth": "Approximately 228,000 people were employed on a full-time basis as of June 30, 2024: 126,000 in the U.S. and 102,000 internationally.",
    },
    {
        "question": "Who is the company's independent registered public accounting firm?",
        "ground_truth": "Deloitte & Touche LLP, which has served as the company's auditor since 1983.",
    },
    {
        "question": "What are the main risk factors disclosed in the filing?",
        "ground_truth": "Risk factors include intense competition across all markets, competition among platform-based ecosystems, cybersecurity and data privacy risks including cyberattacks and security vulnerabilities, and risks relating to the development and deployment of AI.",
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
        contexts = [rag.doc_to_context(doc) for doc in result.retrieved_docs]

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
