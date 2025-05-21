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


# # GOOGLE API RESIZING APPROACH
# import os
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# # Read keys
# GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# if not GEMINI_KEY:
#     raise RuntimeError("GEMINI_API_KEY not set in environment")
# if not QDRANT_URL or not QDRANT_API_KEY:
#     raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set in environment")

# # Initialize clients
# genai.configure(api_key=GEMINI_KEY)
# qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# def retrieve_relevant_chunks(query: str, top_k: int = 5):
#     """
#     Embed the query using Gemini with output dimension 768, search Qdrant, and return top-k texts.
#     """
#     # Generate embedding using Gemini experimental model
#     response = genai.embed_content(
#         model="models/gemini-embedding-exp-03-07",
#         content=query,
#         task_type="retrieval_query",
#         output_dimensionality=768
#     )
#     q_emb = response['embedding']
    
#     # Check dimension
#     if len(q_emb) != 768:
#         raise ValueError(f"Embedding dimension mismatch: expected 768, got {len(q_emb)}")
    
#     # Perform vector search
#     hits = qdrant.search(
#         collection_name="legal_docs",
#         query_vector=q_emb,
#         limit=top_k
#     )
#     # Return texts
#     return [hit.payload["text"] for hit in hits]

# ==================================

# FINAL MAIN IMPLEMENATION
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

if not api_key or not qdrant_url or not qdrant_key:
    raise RuntimeError("Missing GEMINI_API_KEY, QDRANT_URL, or QDRANT_API_KEY")

# Configure Gen AI client (for embeddings)
# genai.configure(api_key=api_key)
# client = genai.Client()
client = genai.Client(api_key=api_key)

# Initialize Qdrant
qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[str]:
    # 1. Call the Gemini embedding endpoint
    resp = client.models.embed_content(
        # model="gemini-embedding-exp-03-07",
        model="models/text-embedding-004",
        contents=query,
    )

    # 2. Unwrap into a pure Python list of floats:
    #    .embeddings is always a list[ContentEmbedding]
    embeddings_list = resp.embeddings
    if not isinstance(embeddings_list, list) or not embeddings_list:
        raise RuntimeError(f"Unexpected embeddings format: {type(resp.embeddings)}")
    # Take the first embedding vector
    first_embedding = embeddings_list[0]
    if not hasattr(first_embedding, "values"):
        raise RuntimeError(f"ContentEmbedding missing .values: {first_embedding}")
    q_emb: list[float] = first_embedding.values

    # 3. Perform vector search with raw floats
    hits = qdrant.search(
        collection_name="legal_docs",
        query_vector=q_emb,
        limit=top_k
    )
    return [hit.payload["text"] for hit in hits]