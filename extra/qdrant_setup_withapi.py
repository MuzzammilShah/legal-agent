# Used Gemini api for the embedding, but hit too many error limits.

import os
import time
from dotenv import load_dotenv

# 1. LangChain loader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 2. Google Gen AI SDK
from google import genai

# 3. Qdrant client & models
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize clients
genai_client = genai.Client(api_key=GEMINI_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create collection if needed
if not qdrant.collection_exists("legal_docs"):
    qdrant.create_collection(
        collection_name="legal_docs",
        vectors_config=models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE
        )
    )

# Load & split documents
loaders = [
    PyPDFLoader("data/Guide_to_Litigation_India.pdf"),
    PyPDFLoader("data/Legal_Compliance_ICAI.pdf")
]
docs = [doc for loader in loaders for doc in loader.load()]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# # Embed chunks in batches (to avoid rate limits)
# texts = [chunk.page_content for chunk in chunks]
# batch_size = 50
# points = []
# for i in range(0, len(texts), batch_size):
#     batch = texts[i : i + batch_size]
#     resp = genai_client.models.embed_content(
#         model="gemini-embedding-exp-03-07",
#         contents=batch
#     )
#     for idx, emb in enumerate(resp.embeddings, start=i):
#         points.append({
#             "id": idx,
#             "vector": emb,
#             "payload": {"text": texts[idx]}
#         })

# # Upsert into Qdrant
# qdrant.upsert(
#     collection_name="legal_docs",
#     points=points
# )

# print(f"Uploaded {len(points)} embeddings to Qdrant.")

# Embed chunks individually to comply with rate limits
texts = [chunk.page_content for chunk in chunks]
points = []

for idx, text in enumerate(texts):
    try:
        resp = genai_client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text
        )
        embedding = resp.embeddings
        points.append({
            "id": idx,
            "vector": embedding,
            "payload": {"text": text}
        })
        print(f"Embedded chunk {idx + 1}/{len(texts)}")
    except Exception as e:
        print(f"Error embedding chunk {idx + 1}: {e}")
    
    # Introduce a delay to respect rate limits
    time.sleep(5)  # 5-second delay between requests

# Upsert into Qdrant
qdrant.upsert(
    collection_name="legal_docs",
    points=points
)

print(f"Uploaded {len(points)} embeddings to Qdrant.")