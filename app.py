import streamlit as st
from dotenv import load_dotenv
from src.indexing import build_vectorstore_from_transcript, build_vectorstore
from src.chain import build_chain
from pinecone import Pinecone
from src.utils import extract_video_id

load_dotenv()

st.set_page_config(page_title="YouTube RAG", layout="centered")
st.title("YouTube Video Q&A")
st.caption("Ask questions about any YouTube video with an English transcript")


# def extract_video_id(url: str) -> str:
#     url = url.strip()
#     if "v=" in url:
#         return url.split("v=")[1].split("&")[0]
#     elif "youtu.be/" in url:
#         return url.split("youtu.be/")[1].split("?")[0]
#     elif url.isalnum() and len(url) == 11:
#         return url
#     return None


def is_indexed(video_id: str) -> bool:
    try:
        pc = Pinecone()
        index = pc.Index("youtube-rag")
        stats = index.describe_index_stats()
        return video_id in stats.namespaces
    except Exception:
        return False


# --- Stage 1: Video Loading ---
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
            st.session_state.messages = []
            st.session_state.pop("transcript_needed", None)

            if is_indexed(video_id):
                st.session_state.chain = build_chain(video_id=video_id)
                st.success("Video already indexed. Ready to chat.")
            else:
                with st.spinner("Fetching transcript and indexing video..."):
                    try:
                        build_vectorstore(video_id)
                        st.session_state.chain = build_chain(video_id=video_id)
                        st.success("Video indexed. Ready to chat.")
                    except RuntimeError:
                        # Auto fetch failed, ask user for transcript
                        st.session_state.transcript_needed = True
                        st.warning("Could not fetch transcript automatically. Please paste it below.")
                    
                    # # Temporary: force fallback for testing
                    # st.session_state.transcript_needed = True

# --- Fallback: Manual transcript input ---
if st.session_state.get("transcript_needed") and "video_id" in st.session_state:
    transcript = st.text_area(
        "Paste the video transcript here",
        height=200,
        placeholder="Copy the transcript from YouTube's transcript panel and paste it here..."
    )

    if st.button("Index Transcript"):
        if not transcript.strip():
            st.warning("Please paste a transcript before indexing.")
        else:
            with st.spinner("Indexing transcript..."):
                try:
                    build_vectorstore_from_transcript(
                        transcript,
                        st.session_state.video_id
                    )
                    st.session_state.chain = build_chain(video_id=st.session_state.video_id)
                    st.session_state.transcript_needed = False
                    st.success("Transcript indexed. Ready to chat.")
                except RuntimeError as e:
                    st.error(f"Something went wrong: {e}")

# --- Stage 2: Chat ---
if "video_id" in st.session_state and not st.session_state.get("transcript_needed"):
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
                response = st.write_stream(st.session_state.chain.stream(question))
                st.session_state.messages.append({"role": "assistant", "content": response})
            except RuntimeError as e:
                st.error(f"Something went wrong while answering: {e}")