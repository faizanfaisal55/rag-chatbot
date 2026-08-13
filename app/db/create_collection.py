from qdrant_client.models import VectorParams, Distance
from app.db.qdrant_connection import client
client.create_collection(
    collection_name="rag_documents",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("Collection created successfully!")