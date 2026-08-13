from app.db.qdrant_connection import client

collections = client.get_collections()

print(collections)