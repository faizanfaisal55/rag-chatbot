import sys
from pathlib import Path

# ==========================================================
# Allow running this script directly (python app/db/search_vectors.py)
# by adding the backend directory to sys.path.
# ==========================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.qdrant_connection import client, COLLECTION_NAME
from app.ingestion.embedder import model


# ==========================================================
# Configuration
# ==========================================================

VECTOR_LIMIT = 10

# Starting point only — NOT final.
# Will be adjusted after inspecting real scores below.
MIN_VECTOR_SIMILARITY = 0.45


# ==========================================================
# Run One Diagnostic Query
# ==========================================================

def run_diagnostic(query: str):

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=VECTOR_LIMIT,
        with_payload=True,
    )

    if not result.points:
        print("No results returned from Qdrant.")
        return

    for rank, point in enumerate(result.points, start=1):

        payload = point.payload or {}

        score = float(getattr(point, "score", 0.0))
        source = payload.get("source", "Unknown")
        page = payload.get("page_number", "N/A")
        chunk_id = payload.get("chunk_id", point.id)
        text_preview = str(payload.get("text", ""))[:80].replace("\n", " ")

        relevance = "RELEVANT" if score >= MIN_VECTOR_SIMILARITY else "below threshold"

        print(
            f"[{rank}] score={score:.4f} | {relevance:<15} | "
            f"source={source} | page={page} | chunk_id={chunk_id}"
        )
        print(f"     text: {text_preview}...")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    test_queries = [
        "Does Faizan know Python?",          # clearly relevant question
        "What is the weather today?",         # unrelated question
        "Tell me about a person named Zeeshan who is not in any document",  # absent-person question
    ]

    for query in test_queries:
        run_diagnostic(query)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(
        f"Current MIN_VECTOR_SIMILARITY = {MIN_VECTOR_SIMILARITY} "
        "(placeholder — adjust based on scores above)"
    )