import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()
client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)

# List all collections
collections = client.get_collections().collections
print("Collections:", [c.name for c in collections])

# Count all points in your collection (exact count)
count_response = client.count(collection_name="legal_docs")
print("Total points in 'legal_docs':", count_response.count)

# Paginate through the first 5 points
points, next_offset = client.scroll(
    collection_name="legal_docs",
    limit=5
)

if points:
    for pt in points:
        print(f"ID={pt.id}, Text excerpt={pt.payload['text'][:80]}…")
else:
    print("⚠️ No points returned by scroll(), collection may be empty.")