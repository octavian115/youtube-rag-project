from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

VIDEO_ID = "Gfr50f6ZBvo"
FAISS_PATH = f"vectorstore/{VIDEO_ID}"

print(f"Loading FAISS index from {FAISS_PATH}...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
faiss_store = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Extract documents from FAISS
docs = list(faiss_store.docstore._dict.values())
print(f"Found {len(docs)} chunks, pushing to Pinecone...")

# Push to Pinecone
PineconeVectorStore.from_documents(
    docs,
    embeddings,
    index_name="youtube-rag",
    namespace=VIDEO_ID
)

print(f"Done. {len(docs)} chunks pushed to Pinecone namespace: {VIDEO_ID}")