import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import re

from app.services.rag import (
    _vector_search,
    _load_all_documents,
    _select_target_source,
    _filter_by_source,
    _extract_query_keywords,
)
from app.retrieval.hybrid_search import HybridRetriever
from evaluation.test_questions import TEST_QUESTIONS


# ==========================================================
# Keyword presence check
#
# Checks whether the top-ranked chunk actually contains the
# query's meaningful keywords. This measures whether the
# retrieval method surfaces chunks with exact keyword/phrase
# matches — the specific thing BM25 is meant to improve over
# pure vector similarity.
# ==========================================================

def keyword_hit_rate(chunk_text, keywords):

    if not keywords:
        return 0.0

    text_lower = chunk_text.lower()

    hits = sum(1 for kw in keywords if kw in text_lower)

    return hits / len(keywords)


# ==========================================================
# Vector-only ranking (no BM25, no RRF)
# ==========================================================

def rank_vector_only(vector_results, top_k=3):

    ranked = sorted(
        vector_results,
        key=lambda d: d.get("similarity", 0.0),
        reverse=True,
    )

    return ranked[:top_k]


# ==========================================================
# Main comparison
# ==========================================================

def main():

    print("\n" + "=" * 70)
    print("HYBRID SEARCH COMPARISON (vector-only vs vector+BM25/RRF)")
    print("=" * 70)

    all_documents = _load_all_documents()

    print(f"\nTotal chunks in knowledge base: {len(all_documents)}")

    vector_only_scores = []
    hybrid_scores = []

    positive_tests = [
        t for t in TEST_QUESTIONS if t.get("expected_source")
    ]

    print(f"Evaluating on {len(positive_tests)} positive (document-grounded) questions.\n")

    for test in positive_tests:

        question = test["question"]
        expected_source = test["expected_source"]

        print("-" * 70)
        print(f"Question: {question}")

        keywords = _extract_query_keywords(question)

        vector_results = _vector_search(question, limit=30)

        if not vector_results:
            print("  No vector results, skipping.")
            continue

        selected_source = _select_target_source(question, vector_results)

        if not selected_source:
            print("  No source selected, skipping.")
            continue

        isolated_vector_results = _filter_by_source(vector_results, selected_source)
        isolated_documents = _filter_by_source(all_documents, selected_source)

        # ------------------------------------------------
        # Vector-only top chunk
        # ------------------------------------------------

        vector_top = rank_vector_only(isolated_vector_results, top_k=1)
        vector_top_text = vector_top[0]["text"] if vector_top else ""
        vector_hit = keyword_hit_rate(vector_top_text, keywords)
        vector_only_scores.append(vector_hit)

        # ------------------------------------------------
        # Hybrid top chunk
        # ------------------------------------------------

        hybrid = HybridRetriever(
            documents=isolated_documents,
            vector_results=isolated_vector_results,
        )

        hybrid_results = hybrid.search(query=question, top_k=1)
        hybrid_top_text = hybrid_results[0]["text"] if hybrid_results else ""
        hybrid_hit = keyword_hit_rate(hybrid_top_text, keywords)
        hybrid_scores.append(hybrid_hit)

        print(f"  Keywords              : {keywords}")
        print(f"  Vector-only keyword hit rate : {vector_hit:.2f}")
        print(f"  Hybrid keyword hit rate      : {hybrid_hit:.2f}")

    avg_vector = sum(vector_only_scores) / len(vector_only_scores) if vector_only_scores else 0
    avg_hybrid = sum(hybrid_scores) / len(hybrid_scores) if hybrid_scores else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nAverage keyword hit rate (top-1 chunk):")
    print(f"  Vector-only : {avg_vector:.2%}")
    print(f"  Hybrid      : {avg_hybrid:.2%}")

    if avg_hybrid > avg_vector:
        print(f"\nHybrid search improved top-chunk keyword coverage by "
              f"{(avg_hybrid - avg_vector):.2%} over vector-only search.")
    elif avg_hybrid < avg_vector:
        print(f"\nVector-only performed better on this test set by "
              f"{(avg_vector - avg_hybrid):.2%}.")
    else:
        print("\nNo difference observed on this test set.")

    print("=" * 70)


if __name__ == "__main__":
    main()