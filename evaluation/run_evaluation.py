import sys
from pathlib import Path

# ==========================================================
# Make backend directory available for imports
# ==========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ==========================================================
# RAG imports
# ==========================================================

from app.services.rag import (
    _vector_search,
    _load_all_documents,
    _select_target_source,
    _has_keyword_match,
    MIN_VECTOR_SIMILARITY_NO_KEYWORD,
)

from app.retrieval.hybrid_search import HybridRetriever

from evaluation.test_questions import TEST_QUESTIONS


# ==========================================================
# Source normalization
# ==========================================================

def normalize_source(source):
    if not source:
        return ""

    source = str(source).strip().replace("\\", "/")
    source = source.split("/")[-1]

    while source.lower().endswith(".pdf.pdf"):
        source = source[:-4]

    return source.lower()


# ==========================================================
# Evaluation
# ==========================================================

def run_evaluation():

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Load complete knowledge base
    # ------------------------------------------------------

    print("\nLoading documents...")

    all_documents = _load_all_documents()

    print(
        f"Total chunks in knowledge base: "
        f"{len(all_documents)}"
    )

    total = len(TEST_QUESTIONS)
    correct = 0

    # ======================================================
    # Run test questions
    # ======================================================

    for index, test in enumerate(
        TEST_QUESTIONS,
        start=1
    ):

        question = test["question"]
        expected = test.get(
            "expected_source"
        )

        print("\n" + "-" * 70)
        print(
            f"[{index}/{total}] {question}"
        )

        print(
            "Expected source:",
            expected or "No document"
        )

        # --------------------------------------------------
        # Vector search
        # --------------------------------------------------

        vector_results = _vector_search(
            question,
            limit=30
        )

        top_score = max(
            (doc.get("similarity", 0.0) for doc in vector_results),
            default=0.0,
        )

        keyword_match = _has_keyword_match(
            question,
            vector_results,
        )

        print(
            f"Top similarity score: {top_score:.4f} "
            f"| keyword match: {keyword_match}"
        )

        if not vector_results or (
            not keyword_match
            and top_score < MIN_VECTOR_SIMILARITY_NO_KEYWORD
        ):

            print("Vector results: NONE (no keyword match, below threshold)")

            actual = None

        else:

            print(
                f"Vector results: "
                f"{len(vector_results)}"
            )

            # --------------------------------------------------
            # Select target document
            # --------------------------------------------------

            selected_source = _select_target_source(
                question,
                vector_results
            )

            actual = selected_source

            print(
                "Selected source:",
                selected_source or "None"
            )

            # --------------------------------------------------
            # Hybrid retrieval
            # --------------------------------------------------

            if selected_source:

                hybrid = HybridRetriever(
                    documents=all_documents,
                    vector_results=vector_results,
                    allowed_sources=[
                        selected_source
                    ],
                )

                results = hybrid.search(
                    query=question,
                    top_k=6
                )

                if results:

                    actual = results[0].get(
                        "source"
                    )

                    print(
                        "Hybrid top source:",
                        actual
                    )

                else:

                    print(
                        "Hybrid results: NONE"
                    )

        # ==================================================
        # Compare expected vs actual
        # ==================================================

        expected_normalized = normalize_source(
            expected
        )

        actual_normalized = normalize_source(
            actual
        )

        # --------------------------------------------------
        # Positive test
        # --------------------------------------------------

        if expected:

            passed = (
                actual_normalized
                == expected_normalized
            )

        # --------------------------------------------------
        # Negative test
        # --------------------------------------------------

        else:

            passed = actual is None

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        if passed:

            correct += 1

            print("RESULT: PASS")

        else:

            print("RESULT: FAIL")

        print(
            "Expected:",
            expected or "No document"
        )

        print(
            "Actual  :",
            actual or "No document"
        )

    # ======================================================
    # Final accuracy
    # ======================================================

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Retrieval Accuracy: "
        f"{accuracy:.2f}%"
    )

    print("=" * 70)


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    run_evaluation()