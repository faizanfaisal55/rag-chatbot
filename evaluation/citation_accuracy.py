import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.rag import ask_rag
from evaluation.test_questions import TEST_QUESTIONS


def main():

    print("\n" + "=" * 70)
    print("CITATION ACCURACY EVALUATION")
    print("=" * 70)
    print(
        "\nFor each answer, you'll be shown the generated answer "
        "and its cited source(s)."
    )
    print(
        "Manually judge: does the cited source actually support "
        "(contain evidence for) the claims made in the answer?"
    )
    print("Type 'y' if yes, 'n' if no, then press Enter.\n")

    positive_tests = [
        t for t in TEST_QUESTIONS if t.get("expected_source")
    ]

    results = []

    for index, test in enumerate(positive_tests, start=1):

        question = test["question"]

        print("-" * 70)
        print(f"[{index}/{len(positive_tests)}] Question: {question}")

        result = ask_rag(question, history=[])

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        print(f"\nAnswer:\n{answer}")

        print(f"\nCited sources:")
        if sources:
            for source in sources:
                print(
                    f"  - {source.get('source')} "
                    f"(Page {source.get('page_number')})"
                )
        else:
            print("  (none)")

        judgment = input(
            "\nDoes the cited source support this answer? (y/n): "
        ).strip().lower()

        supported = judgment == "y"

        results.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "supported": supported,
        })

        print()

    # ======================================================
    # Summary
    # ======================================================

    total = len(results)
    supported_count = sum(1 for r in results if r["supported"])
    accuracy = (supported_count / total * 100) if total else 0

    print("\n" + "=" * 70)
    print("CITATION ACCURACY RESULTS")
    print("=" * 70)

    for r in results:
        status = "✅ SUPPORTED" if r["supported"] else "❌ NOT SUPPORTED"
        print(f"{status} — {r['question']}")

    print(
        f"\nCitation accuracy: {supported_count}/{total} "
        f"({accuracy:.2f}%)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()