import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def fetch_transcript(video_id: str) -> str:
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=['en'])
        return " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        raise ValueError(f"Transcripts are disabled for video: {video_id}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching transcript: {e}")


def build_vectorstore(video_id: str, save_path: str = "vectorstore") -> FAISS:
    if not video_id or not video_id.strip():
        raise ValueError("Video ID cannot be empty")

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

    if not chunks:
        raise ValueError("No chunks generated from transcript")

    # Step 3 - Embed and store
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {e}")


    # Step 4 - Persist
    try:
        os.makedirs(save_path, exist_ok=True)
        vector_store.save_local(save_path)
        print(f"Vectorstore saved to {save_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save vectorstore: {e}")

    return vector_store