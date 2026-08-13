import hashlib

from qdrant_client.models import PointStruct

from app.db.qdrant_connection import client, COLLECTION_NAME
from app.ingestion.embedder import generate_embeddings


def create_point_id(source, chunk_id):
    """
    Create a stable unique ID for each document chunk.
    """

    value = f"{source}::{chunk_id}"

    hash_value = hashlib.md5(
        value.encode("utf-8")
    ).hexdigest()

    return int(hash_value[:15], 16)


def store_vectors(chunks, source):

    if not chunks:
        return 0

    # --------------------------------------------------
    # Extract text from chunks
    # --------------------------------------------------

    valid_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("text")
    ]

    if not valid_chunks:
        return 0

    texts = [
        chunk["text"]
        for chunk in valid_chunks
    ]

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    embeddings = generate_embeddings(texts)

    points = []

    # --------------------------------------------------
    # Create Qdrant points
    # --------------------------------------------------

    for i, (chunk, embedding) in enumerate(
        zip(valid_chunks, embeddings)
    ):

        point_id = create_point_id(
            source,
            i
        )

        points.append(
            PointStruct(
                id=point_id,

                vector=embedding.tolist(),

                payload={
                    "text": chunk["text"],

                    # PDF filename
                    "source": source,

                    # Page number
                    "page_number": chunk.get(
                        "page_number",
                        "Unknown"
                    ),

                    # Chunk number
                    "chunk_id": i,

                    # Used to identify the document
                    "document_id": source
                }
            )
        )

    # --------------------------------------------------
    # Upload vectors to Qdrant
    # --------------------------------------------------

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("=" * 60)
    print("VECTORS STORED")
    print("=" * 60)
    print("Document:", source)
    print("Vectors:", len(points))
    print("=" * 60)

    return len(points)