import streamlit as st
from dotenv import load_dotenv
from src.indexing import build_vectorstore
from src.chain import build_chain
from pinecone import Pinecone
# import os

load_dotenv()

st.set_page_config(page_title="YouTube RAG", layout="centered")
st.title("YouTube Video Q&A")
st.caption("Ask questions about any YouTube video with an English transcript")


def extract_video_id(url: str) -> str:
    url = url.strip()
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif url.isalnum() and len(url) == 11:
        return url  # raw video ID
    return None


def is_indexed(video_id: str) -> bool:
    try:
        pc = Pinecone()
        index = pc.Index("youtube-rag")
        stats = index.describe_index_stats()
        return video_id in stats.namespaces
    except Exception:
        return False


# Stage 1: Video Loading
url = st.text_input("Paste a YouTube URL")

if st.button("Load Video"):
    if not url:
        st.warning("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(url)

        if not video_id:
            st.error("Invalid YouTube URL. Please paste a valid video link.")
        else:
            st.session_state.video_id = video_id
            st.session_state.messages = [] # to clear chat history on new video

            if is_indexed(video_id):
                st.success("Video already indexed. Ready to chat.")
                st.session_state.chain = build_chain(video_id=video_id)
            else:
                with st.spinner("Fetching transcript and indexing video..."):
                    try:
                        build_vectorstore(video_id)
                        #this builds the chain once per video load and not per query
                        st.session_state.chain = build_chain(video_id=video_id)
                        st.success("Video indexed. Ready to chat.")
                    except ValueError as e:
                        st.error(f"{e}")
                        st.session_state.pop("video_id", None)
                    except RuntimeError as e:
                        st.error(f"Something went wrong: {e}")
                        st.session_state.pop("video_id", None)

# Stage 2: Chat
if "video_id" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about the video"):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                chain = st.session_state.chain
                response = st.write_stream(chain.stream(question))
                st.session_state.messages.append({"role": "assistant", "content": response})
            except FileNotFoundError as e:
                st.error(f"{e}")
            except RuntimeError as e:
                st.error(f"Something went wrong while answering: {e}")