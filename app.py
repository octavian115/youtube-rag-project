import streamlit as st
from dotenv import load_dotenv
from src.indexing import build_vectorstore
from src.chain import build_chain
import os

load_dotenv()

st.set_page_config(page_title="YouTube RAG", layout="centered")
st.title("YouTube Video Q&A")
st.caption("Ask questions about any YouTube video")


def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url  # assume raw video id was passed


def is_indexed(video_id: str) -> bool:
    return os.path.exists(f"vectorstore/{video_id}")


# Stage 1: Video Loading 
url = st.text_input("Paste a YouTube URL")

if st.button("Load Video"):
    if not url:
        st.warning("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(url)
        st.session_state.video_id = video_id

        if is_indexed(video_id):
            st.success("Video already indexed. Ready to chat.")
        else:
            with st.spinner("Indexing video..."):
                try:
                    build_vectorstore(video_id, save_path=f"vectorstore/{video_id}")
                    st.success("Video indexed. Ready to chat.")
                except Exception as e:
                    st.error(f"Failed to index video: {e}")

# Stage 2: Chat
if "video_id" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question about the video"):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            chain = build_chain(save_path=f"vectorstore/{st.session_state.video_id}")
            response = st.write_stream(chain.stream(question))

        st.session_state.messages.append({"role": "assistant", "content": response})