import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from google import genai

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
genai_client = genai.Client(api_key=GEMINI_KEY)

def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """Embed query, search Qdrant, return top-k texts."""
    q_emb = genai_client.embeddings.create(
        model="gemini-2.0-text-embedding",
        instances=[query]
    ).embeddings[0]
    hits = qdrant.search(
        collection_name="legal_docs",
        query_vector=q_emb,
        limit=top_k
    )
    return [hit.payload["text"] for hit in hits]