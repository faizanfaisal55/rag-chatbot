import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.hallucination_check import check_groundedness


def print_result(label, result):
    print(f"\n{label}")
    print(f"  Checked              : {result['checked']}")
    print(f"  Groundedness score   : {result['groundedness_score']}")
    print(f"  Flagged as hallucination: {result['flagged']}")
    if result["claims_missing"]:
        print(f"  Claims NOT found in context: {result['claims_missing']}")


def main():

    print("\n" + "=" * 70)
    print("HALLUCINATION CHECK — DEMONSTRATION")
    print("=" * 70)

    # This is real retrieved context, matching what the RAG
    # pipeline would actually retrieve for Umair.pdf.
    context = """
    [SOURCE 1]
    Document: Umair.pdf
    Page: 1
    Text:
    Muhammad Umair is a former employee of M/S. Hillcrest Solutions
    Private Limited. He severed his employment and all connections
    with the company and its client, M/S. U Microfinance Bank
    Limited, after receiving Rs. 58,225 as a full and final
    settlement of all his legal and statutory dues.
    """

    # ------------------------------------------------------
    # Example 1: Correctly grounded answer
    #
    # Every specific fact here (name, company, amount) is
    # taken directly from the context above.
    # ------------------------------------------------------

    grounded_answer = (
        "Muhammad Umair is a former employee of Hillcrest "
        "Solutions Private Limited. He received Rs. 58,225 as "
        "a full and final settlement."
    )

    result_grounded = check_groundedness(grounded_answer, context)
    print_result("Example 1 — Grounded answer (should NOT be flagged):", result_grounded)

    # ------------------------------------------------------
    # Example 2: Hallucinated answer
    #
    # This answer invents a different settlement amount and
    # a company name never mentioned in the source — the kind
    # of fabrication an ungrounded LLM response can produce.
    # ------------------------------------------------------

    hallucinated_answer = (
        "Muhammad Umair is a former employee of Zenith Corp. "
        "He received Rs. 250,000 as a full and final settlement."
    )

    result_hallucinated = check_groundedness(hallucinated_answer, context)
    print_result("Example 2 — Hallucinated answer (SHOULD be flagged):", result_hallucinated)

    # ------------------------------------------------------
    # Example 3: Refusal answer (should be skipped, not flagged)
    # ------------------------------------------------------

    refusal_answer = "I couldn't find that information in the uploaded documents."

    result_refusal = check_groundedness(refusal_answer, context)
    print_result("Example 3 — Refusal answer (should be skipped):", result_refusal)

    print("\n" + "=" * 70)

    if result_grounded["flagged"]:
        print("❌ UNEXPECTED: grounded answer was flagged.")
    else:
        print("✅ Grounded answer correctly passed.")

    if result_hallucinated["flagged"]:
        print("✅ Hallucinated answer correctly caught.")
    else:
        print("❌ UNEXPECTED: hallucinated answer was NOT caught.")

    print("=" * 70)


if __name__ == "__main__":
    main()