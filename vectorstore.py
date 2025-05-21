# # OPEN SOURCE APPROACH
# import os
# import numpy as np
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# import ollama

# # Load environment variables
# load_dotenv()

# # Read keys and configurations
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Default to local if not set

# if not QDRANT_URL or not QDRANT_API_KEY:
#     raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set in environment")

# # Initialize clients
# qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
# ollama_client = ollama.Client(host=OLLAMA_HOST)

# def retrieve_relevant_chunks(query: str, top_k: int = 5):
#     """
#     Embed the query using Ollama, search Qdrant, and return top-k texts.
#     """
#     # Generate embedding using Ollama
#     resp = ollama_client.embed(model="nomic-embed-text", input=query)
#     q_emb = resp.get("embeddings") or resp.get("embedding")
#     if q_emb is None:
#         raise ValueError("Missing embedding for query")
#     q_emb = np.array(q_emb).flatten().tolist()
#     if len(q_emb) != 768:
#         raise ValueError(f"Query embedding dimension mismatch: got {len(q_emb)}")

#     # Perform vector search
#     hits = qdrant.search(
#         collection_name="legal_docs",
#         query_vector=q_emb,
#         limit=top_k
#     )
#     # Return texts
#     return [hit.payload["text"] for hit in hits]


# GOOGLE API RESIZING APPROACH
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Read keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set in environment")

# Initialize clients
genai.configure(api_key=GEMINI_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """
    Embed the query using Gemini with output dimension 768, search Qdrant, and return top-k texts.
    """
    # Generate embedding using Gemini experimental model
    response = genai.embed_content(
        model="models/gemini-embedding-exp-03-07",
        content=query,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    q_emb = response['embedding']
    
    # Check dimension
    if len(q_emb) != 768:
        raise ValueError(f"Embedding dimension mismatch: expected 768, got {len(q_emb)}")
    
    # Perform vector search
    hits = qdrant.search(
        collection_name="legal_docs",
        query_vector=q_emb,
        limit=top_k
    )
    # Return texts
    return [hit.payload["text"] for hit in hits]