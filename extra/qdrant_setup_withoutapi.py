from qdrant_client import QdrantClient
from qdrant_client.client_base import QdrantBase
import ollama
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http import models
import os

# 1. Load env & init Qdrant
load_dotenv()
QDRANT_URL    = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 2. (Re)create collection with 768-dim vectors if needed
if qdrant.collection_exists("legal_docs"):
    qdrant.delete_collection("legal_docs")
qdrant.create_collection(
    collection_name="legal_docs",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
)

# 3. Load & chunk PDFs
loaders = [
    PyPDFLoader("data/Guide_to_Litigation_India.pdf"),
    PyPDFLoader("data/Legal_Compliance_ICAI.pdf")
]
docs = [d for L in loaders for d in L.load()]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 4. Embed each chunk via Ollama
ids, vectors, payloads = [], [], []
for idx, chunk in enumerate(chunks):
    try:
        resp = ollama.embed(model="nomic-embed-text", input=chunk.page_content)
        vec  = resp.get("embeddings") or resp.get("embedding")
        ids.append(idx)
        vectors.append(vec)
        payloads.append({"text": chunk.page_content})
        print(f"✅ Embedded chunk {idx+1}/{len(chunks)}")
    except Exception as e:
        print(f"❌ Error embedding chunk {idx+1}: {e}")

# 5. Upload all at once, letting the client choose the proper schema

# Option A: upload_points()
# qdrant.upload_points(
#     collection_name="legal_docs",
#     points=[{"id": i, "vector": vec, "payload": {"text": text}}
#             for i, (vec, text) in enumerate(zip(vectors, payloads))],
#     batch_size=128
# )
# print("✅ Uploaded via upload_points()")

# Option B: upload_collection()
qdrant.upload_collection(
    collection_name="legal_docs",
    ids=ids,
    vectors=vectors,
    payloads=payloads,
    batch_size=128
)
print("✅ Uploaded via upload_collection()")