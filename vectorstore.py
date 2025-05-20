# # vectorstore.py (Approach C)
# import os
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# # 1) Load environment
# load_dotenv()

# # 2) Read keys
# GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# if not GEMINI_KEY:
#     raise RuntimeError("GEMINI_API_KEY not set in environment")
# if not QDRANT_URL or not QDRANT_API_KEY:
#     raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set in environment")

# # 3) Initialize clients
# qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# # 4) LangChain embedding wrapper
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-exp-03-07",  # same experimental model :contentReference[oaicite:2]{index=2}
#     google_api_key=GEMINI_KEY
# )

# def retrieve_relevant_chunks(query: str, top_k: int = 5):
#     """
#     Embed the query via LangChain helper, search Qdrant, return top-k texts.
#     """
#     # 5) Generate embedding
#     q_emb = embeddings.embed_query(query)  # returns a list of floats :contentReference[oaicite:3]{index=3}

#     # 6) Perform vector search
#     hits = qdrant.search(
#         collection_name="legal_docs",
#         query_vector=q_emb,
#         limit=top_k
#     )
#     # 7) Return texts
#     return [hit.payload["text"] for hit in hits]

import os
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import ollama

# Load environment variables
load_dotenv()

# Read keys and configurations
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Default to local if not set

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set in environment")

# Initialize clients
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
ollama_client = ollama.Client(host=OLLAMA_HOST)

def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """
    Embed the query using Ollama, search Qdrant, and return top-k texts.
    """
    # Generate embedding using Ollama
    resp = ollama_client.embed(model="nomic-embed-text", input=query)
    q_emb = resp.get("embeddings") or resp.get("embedding")
    if q_emb is None:
        raise ValueError("Missing embedding for query")
    q_emb = np.array(q_emb).flatten().tolist()
    if len(q_emb) != 768:
        raise ValueError(f"Query embedding dimension mismatch: got {len(q_emb)}")

    # Perform vector search
    hits = qdrant.search(
        collection_name="legal_docs",
        query_vector=q_emb,
        limit=top_k
    )
    # Return texts
    return [hit.payload["text"] for hit in hits]