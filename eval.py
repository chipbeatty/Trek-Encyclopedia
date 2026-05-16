"""
eval.py
Evaluates retrieval and generation quality against a golden Q&A set.

Retrieval metrics:
  Recall@K  — fraction of questions where a correct episode appears in top K results
  MRR       — mean reciprocal rank of the first correct result

Generation metric:
  Faithfulness — fraction of answers containing only claims supported by retrieved chunks
                 (checked via a second gpt-4o-mini call)

Usage:
    uv run python eval.py                        # retrieval only
    uv run python eval.py --faithfulness         # retrieval + generation
    uv run python eval.py --golden data/golden_qa.json --k 1 3 5 10
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI

from rag import ask, build_context
from search import search

_openai = OpenAI()


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def _retrieval_results(question: str, top_k: int) -> tuple[list[str], float]:
    """Return (returned_episode_ids, reciprocal_rank) for a question."""
    results = search(question, top_k=top_k)
    return [r["metadata"]["episode_id"] for r in results], results


def run_retrieval_eval(golden: list[dict], k_values: list[int]) -> None:
    max_k = max(k_values)
    hits: dict[int, int] = {k: 0 for k in k_values}
    rr_sum = 0.0
    misses: list[dict] = []

    print(f"── Retrieval  ({len(golden)} questions, top_k={max_k}) {'─' * 20}")

    for item in golden:
        question = item["question"]
        expected_ids = set(item["episode_ids"])

        returned_ids, _ = _retrieval_results(question, top_k=max_k)

        rr = 0.0
        for rank, eid in enumerate(returned_ids, 1):
            if eid in expected_ids:
                rr = 1.0 / rank
                break
        rr_sum += rr

        hit_ks = {k: any(eid in expected_ids for eid in returned_ids[:k]) for k in k_values}
        for k in k_values:
            if hit_ks[k]:
                hits[k] += 1

        marker = "✓" if hit_ks[max_k] else "✗"
        print(f"  {marker}  {question[:72]}")

        if not hit_ks[max_k]:
            misses.append({
                "question": question,
                "expected": item["episode_ids"],
                "returned": returned_ids[:5],
            })

    n = len(golden)
    print(f"\n  {'Metric':<12} {'Score':>10}")
    print(f"  {'─' * 24}")
    for k in k_values:
        print(f"  Recall@{k:<5}  {hits[k]:2d}/{n}  ({100 * hits[k] / n:.1f}%)")
    print(f"  MRR          {rr_sum / n:.3f}")

    if misses:
        print(f"\n  Misses at K={max_k}:")
        for m in misses:
            print(f"    Q: {m['question'][:70]}")
            print(f"       Expected : {m['expected']}")
            print(f"       Returned : {m['returned']}")
    else:
        print(f"\n  Perfect recall at K={max_k}!")


# ---------------------------------------------------------------------------
# Faithfulness evaluation
# ---------------------------------------------------------------------------

def check_faithfulness(answer: str, chunks: list[dict]) -> dict:
    """
    Ask gpt-4o-mini whether every claim in the answer is supported by
    the source chunks. Returns {"faithful": bool, "unsupported_claims": list[str]}.
    """
    context = build_context(chunks)
    response = _openai.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You evaluate whether an AI-generated answer is grounded in provided "
                    "source excerpts. Respond only in JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Source excerpts:\n{context}\n\n"
                    f"Answer to evaluate:\n{answer}\n\n"
                    "List any specific claims in the answer that are NOT directly supported "
                    "by the source excerpts. Episode titles, character names, and plot details "
                    "must appear in the excerpts to count as supported. Ignore hedges like "
                    "'the excerpts suggest' — only flag concrete unsupported facts.\n\n"
                    '{"faithful": true_or_false, "unsupported_claims": ["claim 1", ...]}'
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def run_faithfulness_eval(golden: list[dict]) -> None:
    faithful_count = 0
    unfaithful: list[dict] = []

    print(f"\n── Faithfulness  ({len(golden)} questions) {'─' * 26}")

    for item in golden:
        question = item["question"]
        result = ask(question)
        faith = check_faithfulness(result["answer"], result["chunks"])

        if faith["faithful"]:
            faithful_count += 1
            print(f"  ✓  {question[:72]}")
        else:
            print(f"  ✗  {question[:72]}")
            unfaithful.append({
                "question": question,
                "answer": result["answer"],
                "unsupported_claims": faith["unsupported_claims"],
            })

    n = len(golden)
    print(f"\n  Faithful: {faithful_count}/{n}  ({100 * faithful_count / n:.1f}%)")

    if unfaithful:
        print(f"\n  Unfaithful answers:")
        for u in unfaithful:
            print(f"\n    Q: {u['question']}")
            for claim in u["unsupported_claims"]:
                print(f"       ✗ {claim}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG quality")
    parser.add_argument("--golden", default="data/golden_qa.json")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Values of K for Recall@K (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="Also run faithfulness evaluation (makes LLM calls per question)",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_path}")

    with golden_path.open() as f:
        golden: list[dict] = json.load(f)

    run_retrieval_eval(golden, args.k)

    if args.faithfulness:
        run_faithfulness_eval(golden)

    print()


if __name__ == "__main__":
    main()
