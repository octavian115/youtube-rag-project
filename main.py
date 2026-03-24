from dotenv import load_dotenv
from src.indexing import build_vectorstore
from src.chain import build_chain
import os


load_dotenv()

#only for debugging we're accessing this from app.py
VIDEO_ID = "Gfr50f6ZBvo"
VECTORSTORE_PATH = f"vectorstore/{VIDEO_ID}"


def index(video_id: str = VIDEO_ID):
    print(f"Indexing video: {video_id}")
    build_vectorstore(video_id)
    print("Indexing complete.")


def ask(question: str):
    chain = build_chain(video_id=VIDEO_ID)
    answer = chain.invoke(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")


if __name__ == "__main__":
    # Run indexing only once, comment out after first run
    index()

    # Ask questions
    ask("What did demis say about dangers of AGI?")