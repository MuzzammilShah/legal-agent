# from qdrant_client import QdrantClient
# from qdrant_client.client_base import QdrantBase
# import ollama
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from qdrant_client.http import models
# from qdrant_client.models import PointStruct, Distance, VectorParams
# import os

# # 1. Load env & init Qdrant
# load_dotenv()
# # QDRANT_URL    = os.getenv("QDRANT_URL")
# # QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# # qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
# client = QdrantClient(
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY"),
# )

# # 2. (Re)create collection with 768-dim vectors if needed
# if client.collection_exists("legal_docs"):
#     client.delete_collection("legal_docs")
# client.create_collection(
#     collection_name="legal_docs",
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE),
# )

# # 3. Load & chunk PDFs
# loaders = [
#     PyPDFLoader("data/Guide_to_Litigation_India.pdf"),
#     PyPDFLoader("data/Legal_Compliance_ICAI.pdf")
# ]
# docs = [d for L in loaders for d in L.load()]
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = splitter.split_documents(docs)

# # 4. Embed each chunk via Ollama
# ids, vectors, payloads = [], [], []
# for idx, chunk in enumerate(chunks):
#     try:
#         resp = ollama.embed(model="nomic-embed-text", input=chunk.page_content)
#         vec  = resp.get("embeddings") or resp.get("embedding")
#         ids.append(idx)
#         vectors.append(vec)
#         payloads.append({"text": chunk.page_content})
#         print(f"✅ Embedded chunk {idx+1}/{len(chunks)}")
#     except Exception as e:
#         print(f"❌ Error embedding chunk {idx+1}: {e}")

# # 5. Upload all at once, letting the client choose the proper schema

# points = []
# for idx, chunk in enumerate(chunks):
#     resp = ollama.embed(model="nomic-embed-text", input=chunk.page_content)
#     vec = resp.get("embeddings") or resp.get("embedding")
#     if vec is None:
#         raise ValueError(f"Missing embedding for chunk {idx}")
#     points.append(
#         PointStruct(
#             id=idx,
#             vector=vec,
#             payload={"text": chunk.page_content}
#         )
#     )

# update_info = client.upsert(
#     collection_name="legal_docs",
#     points=points,
#     wait=True
# )
# print("Upsert result:", update_info)

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
import os, numpy as np, ollama
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Init
load_dotenv()
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# 2. (Re)create collection
if client.collection_exists("legal_docs"):
    client.delete_collection("legal_docs")
client.create_collection(
    collection_name="legal_docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# 3. Load & split
loaders = [
    PyPDFLoader("data/Guide_to_Litigation_India.pdf"),
    PyPDFLoader("data/Legal_Compliance_ICAI.pdf"),
]
docs = [d for L in loaders for d in L.load()]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 4. Embed & flatten
points = []
for idx, chunk in enumerate(chunks):
    resp = ollama.embed(model="nomic-embed-text", input=chunk.page_content)
    vec  = resp.get("embeddings") or resp.get("embedding")
    if vec is None:
        raise ValueError(f"Missing embedding for chunk {idx}")
    flat_vec = np.array(vec).flatten().tolist()
    if len(flat_vec) != 768:
        raise ValueError(f"Embedding dimension mismatch: got {len(flat_vec)} floats")
    points.append(PointStruct(id=idx, vector=flat_vec, payload={"text": chunk.page_content}))
    print(f"✅ Prepared point {idx+1}/{len(chunks)}")

# 5. Upsert and wait
result = client.upsert(
    collection_name="legal_docs",
    points=points,
    wait=True,
)
print("✅ Upsert completed:", result)

# 6. Quick verify
cnt = client.count(collection_name="legal_docs").count
print("Total points in 'legal_docs':", cnt)