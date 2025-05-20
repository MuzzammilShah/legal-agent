import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai

# Load secrets
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 1. Connect to Qdrant
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# 2. Create collection if not exists
client.recreate_collection(
    collection_name="legal_docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# 3. Load & split PDFs
loaders = [
    PyPDFLoader("Guide_to_Litigation_India.pdf"),
    PyPDFLoader("Legal_Compliance_ICAI.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 4. Embed chunks
genai_client = genai.Client(api_key=GEMINI_KEY)
texts = [c.page_content for c in chunks]
embeddings = genai_client.embeddings.create(
    model="gemini-2.0-text-embedding",
    instances=texts
).embeddings

# 5. Upload to Qdrant
points = [
    {"id": idx, "vector": emb, "payload": {"text": chunks[idx].page_content}}
    for idx, emb in enumerate(embeddings)
]
client.upsert(
    collection_name="legal_docs",
    points=points
)
print(f"Uploaded {len(points)} embeddings to Qdrant.")