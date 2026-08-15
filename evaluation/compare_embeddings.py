import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import os
import re
import time
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance

from app.db.qdrant_connection import client
from app.ingestion.pdf_loader_fitz import load_pdf_pages
from app.ingestion.text_loader import load_text_file
from app.ingestion.chunker import chunk_pages
from evaluation.test_questions import TEST_QUESTIONS


# ==========================================================
# Config
# ==========================================================

DOCUMENTS_DIR = os.path.join(BACKEND_DIR, "documents")

MODELS_TO_COMPARE = {
    "all-MiniLM-L6-v2": {
        "collection": "embedding_experiment_minilm",
        "vector_size": 384,
    },
    "all-mpnet-base-v2": {
        "collection": "embedding_experiment_mpnet",
        "vector_size": 768,
    },
}

MIN_VECTOR_SIMILARITY_NO_KEYWORD = 0.40


# ==========================================================
# Helpers (mirrors app/services/rag.py logic — kept local so
# this experiment never touches production code)
# ==========================================================

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "who", "what",
    "where", "when", "why", "how", "do", "does", "did", "about",
    "information", "mentioned", "tell", "me", "please", "can",
    "you", "give", "for", "of", "in", "on", "to", "and", "or",
    "this", "that", "him", "her", "his", "their", "they", "it",
    "he", "she", "with", "from", "document", "file",
}


def extract_keywords(query):
    words = re.findall(r"[A-Za-z0-9]+", query.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) >= 2]


def normalize_source(source):
    if not source:
        return ""
    source = str(source).strip().replace("\\", "/").split("/")[-1]
    while source.lower().endswith(".pdf.pdf"):
        source = source[:-4]
    return source.lower()


def has_keyword_match(query, results):
    keywords = extract_keywords(query)
    if not keywords:
        return False
    for r in results:
        source = normalize_source(r["source"])
        filename = source.rsplit(".", 1)[0]
        filename_words = set(re.findall(r"[a-z0-9]+", filename))
        if any(k in filename_words for k in keywords):
            return True
    return False


# ==========================================================
# Build a temporary collection using a given embedding model
# ==========================================================

def build_collection(embedding_model, collection_name, vector_size):

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    point_id = 0

    for filename in os.listdir(DOCUMENTS_DIR):

        file_path = os.path.join(DOCUMENTS_DIR, filename)
        ext = os.path.splitext(filename.lower())[1]

        if ext == ".pdf":
            pages = load_pdf_pages(file_path)
        elif ext in (".txt", ".md"):
            pages = load_text_file(file_path)
        else:
            continue

        if not pages:
            continue

        chunks = chunk_pages(pages)

        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        vectors = embedding_model.encode(
            texts, normalize_embeddings=True
        ).tolist()

        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append({
                "id": point_id,
                "vector": vector,
                "payload": {
                    "text": chunk["text"],
                    "source": filename,
                    "page_number": chunk["page_number"],
                    "chunk_id": point_id,
                }
            })
            point_id += 1

        client.upsert(collection_name=collection_name, points=points)

    return point_id


# ==========================================================
# Run evaluation questions against a collection
# ==========================================================

def evaluate_collection(embedding_model, collection_name):

    correct = 0
    total = len(TEST_QUESTIONS)

    for test in TEST_QUESTIONS:

        question = test["question"]
        expected = test.get("expected_source")

        query_vector = embedding_model.encode(
            question, normalize_embeddings=True
        ).tolist()

        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=30,
            with_payload=True,
        )

        results = [
            {
                "source": p.payload.get("source", "Unknown"),
                "similarity": float(getattr(p, "score", 0.0)),
            }
            for p in search_result.points
        ]

        top_score = max((r["similarity"] for r in results), default=0.0)
        keyword_match = has_keyword_match(question, results)

        if not results or (
            not keyword_match
            and top_score < MIN_VECTOR_SIMILARITY_NO_KEYWORD
        ):
            actual = None
        else:
            source_scores = defaultdict(float)
            keywords = extract_keywords(question)

            for r in results:
                source = normalize_source(r["source"])
                filename = source.rsplit(".", 1)[0]
                filename_words = set(re.findall(r"[a-z0-9]+", filename))
                matches = sum(1 for k in keywords if k in filename_words)
                if matches:
                    source_scores[source] += matches * 20.0

            for rank, r in enumerate(results, start=1):
                source = normalize_source(r["source"])
                source_scores[source] += r["similarity"] * (1.0 / rank)

            actual = max(source_scores, key=source_scores.get) if source_scores else None

        expected_norm = normalize_source(expected)
        actual_norm = normalize_source(actual)

        if expected:
            passed = actual_norm == expected_norm
        else:
            passed = actual is None

        if passed:
            correct += 1

    accuracy = (correct / total * 100) if total else 0
    return correct, total, accuracy


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n" + "=" * 70)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 70)

    results_summary = {}

    for model_name, config in MODELS_TO_COMPARE.items():

        print(f"\n{'-' * 70}")
        print(f"Loading model: {model_name}")
        print("-" * 70)

        load_start = time.time()
        embedding_model = SentenceTransformer(model_name)
        load_time = time.time() - load_start

        print(f"Model loaded in {load_time:.1f}s")

        print(f"Building collection: {config['collection']}...")
        build_start = time.time()
        total_chunks = build_collection(
            embedding_model,
            config["collection"],
            config["vector_size"]
        )
        build_time = time.time() - build_start

        print(f"Stored {total_chunks} chunks in {build_time:.1f}s")

        print(f"Evaluating {model_name}...")
        eval_start = time.time()
        correct, total, accuracy = evaluate_collection(
            embedding_model, config["collection"]
        )
        eval_time = time.time() - eval_start

        results_summary[model_name] = {
            "vector_size": config["vector_size"],
            "chunks": total_chunks,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "build_time": build_time,
            "eval_time": eval_time,
        }

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for model_name, r in results_summary.items():
        print(f"\n{model_name}:")
        print(f"  Vector dimension    : {r['vector_size']}")
        print(f"  Total chunks stored : {r['chunks']}")
        print(f"  Retrieval accuracy  : {r['correct']}/{r['total']} ({r['accuracy']:.2f}%)")
        print(f"  Embedding time      : {r['build_time']:.1f}s")
        print(f"  Query eval time     : {r['eval_time']:.1f}s")

    print("\n" + "=" * 70)

    # Cleanup temporary collections
    for config in MODELS_TO_COMPARE.values():
        client.delete_collection(config["collection"])

    print("Temporary experiment collections cleaned up.")
    print("=" * 70)


if __name__ == "__main__":
    main()