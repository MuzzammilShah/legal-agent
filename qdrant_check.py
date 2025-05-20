# import os, time
# from qdrant_client import QdrantClient
# from dotenv import load_dotenv

# load_dotenv()
# client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# # 1. List collections
# print("Collections:", [c.name for c in client.get_collections().collections])

# # 2. Wait briefly in case of async indexing
# time.sleep(2)

# # 3. Count points
# cnt = client.count(collection_name="legal_docs").count
# print("Total points in 'legal_docs':", cnt)

# # 4. Peek at first 5 points
# pts = client.scroll_points(collection_name="legal_docs", limit=5)
# if pts:
#     for p in pts:
#         print(f"  • ID={p.id}, text[:50]={p.payload['text'][:50]}…")
# else:
#     print("No points found when scrolling—collection empty.")

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