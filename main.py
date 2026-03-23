from dotenv import load_dotenv
from src.indexing import build_vectorstore
from src.chain import build_chain
import os


load_dotenv()

VIDEO_ID = "Gfr50f6ZBvo"
VECTORSTORE_PATH = "vectorstore"


def index(video_id: str = VIDEO_ID):
    print(f"Indexing video: {video_id}")
    build_vectorstore(video_id, save_path=VECTORSTORE_PATH)
    print("Indexing complete.")


def ask(question: str):
    chain = build_chain(save_path=VECTORSTORE_PATH)
    answer = chain.invoke(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")


if __name__ == "__main__":
    # Run indexing only once, comment out after first run
    # index()

    # Ask questions
    # ask("Who is Demis Hassabis?")
    ask("What did demis say about dangers of AGI?")