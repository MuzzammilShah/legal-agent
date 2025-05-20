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