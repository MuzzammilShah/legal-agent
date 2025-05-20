# vectorstore.py (Approach A)
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from google import genai

# 1) Load .env so keys are available
load_dotenv()

# 2) Read your API keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set in environment")

# 3) Initialize clients
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
genai_client = genai.Client(api_key=GEMINI_KEY)

def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """
    Embed the query via Gemini embed_content, search Qdrant, return top-k texts.
    """
    # 4) Generate embeddings with Gemini
    resp = genai_client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=[query],   # <-- just the raw string
    )
    # 5) Pull out the first embedding vector
    q_emb = resp.embeddings[0]

    # 6) Perform vector search in Qdrant
    hits = qdrant.search(
        collection_name="legal_docs",
        query_vector=q_emb,
        limit=top_k
    )
    # 7) Return the raw text payloads
    return [hit.payload["text"] for hit in hits]