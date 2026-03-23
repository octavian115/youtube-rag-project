import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def fetch_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=['en'])
    return " ".join(chunk.text for chunk in transcript_list)


def build_vectorstore(video_id: str, save_path: str = "vectorstore") -> FAISS:
    # Step 1 - Fetch
    transcript = fetch_transcript(video_id)

    # Step 2 - Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents(
        [transcript],
        metadatas=[{"video_id": video_id}]
    )

    # Step 3 - Embed and store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 4 - Persist
    vector_store.save_local(save_path)
    print(f"Vectorstore saved to {save_path}")

    return vector_store