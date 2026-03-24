# YouTube RAG Pipeline

A retrieval-augmented generation (RAG) pipeline that answers questions about a YouTube video using its transcript. Paste a YouTube URL and ask questions in a chat interface.

## Live Demo
[youtube-rag-project.onrender.com](https://youtube-rag-project-n70z.onrender.com)

> The app may take 30-60 seconds to wake up on first visit due to Render's free tier cold starts.

## How it works

1. Fetches the transcript of a YouTube video
2. Splits it into chunks and generates embeddings using OpenAI
3. Stores the embeddings in Pinecone (cloud vector database)
4. On each query, retrieves the most relevant chunks using hybrid search (Pinecone dense + BM25 keyword)
5. Reranks the retrieved chunks using Cohere Rerank
6. Passes the top chunks to GPT-4o-mini to generate a grounded answer

## Project Structure
```
youtube-rag/
├── src/
│   ├── indexing.py      # Transcript fetching, chunking, embedding, storing in Pinecone
│   ├── retrieval.py     # Hybrid search (dense + BM25) and Cohere reranking
│   └── chain.py         # LCEL chain assembly
├── notebooks/           # Experimentation notebooks
├── app.py               # Streamlit UI entry point
├── main.py              # CLI entry point for development and testing
├── migrate_to_pinecone.py  # One-time migration script from FAISS to Pinecone
└── pyproject.toml       # Dependencies
```

## Setup

1. Clone the repo
2. Create a virtual environment and install dependencies
```bash
uv venv && source .venv/bin/activate
uv sync
```
3. Copy `.env.example` to `.env` and add your API keys
```bash
cp .env.example .env
```

## Environment Variables
```
OPENAI_API_KEY=        # Embeddings (text-embedding-3-small) and LLM (gpt-4o-mini)
COHERE_API_KEY=        # Reranking (rerank-english-v3.0)
PINECONE_API_KEY=      # Vector store
```

## Usage

**Run the Streamlit app:**
```bash
streamlit run app.py
```

1. Paste a YouTube URL into the input field
2. Click "Load Video" — the transcript is fetched, chunked, embedded, and stored in Pinecone
3. Ask questions about the video in the chat interface

Previously indexed videos load instantly from Pinecone without re-embedding.

**Run from the CLI (for development and testing):**
```bash
python main.py
```

## Limitations

- Only supports YouTube videos with English transcripts
- Due to YouTube IP blocking on cloud servers, new videos must be indexed locally 
  using `python main.py` and pushed to Pinecone before they can be queried on the live app

## Tech Stack

- [LangChain](https://www.langchain.com/) — LLM framework and LCEL chain
- [Pinecone](https://www.pinecone.io/) — cloud vector store
- [BM25](https://github.com/dorianbrown/rank_bm25) — keyword search
- [Cohere Rerank](https://cohere.com/) — reranking
- [OpenAI](https://openai.com/) — embeddings and LLM
- [Streamlit](https://streamlit.io/) — UI
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — transcript fetching
