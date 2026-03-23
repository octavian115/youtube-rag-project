# YouTube RAG Pipeline

A retrieval-augmented generation (RAG) pipeline that answers questions about a YouTube video using its transcript. Paste a YouTube URL, and ask questions about the video in a chat interface.
Note: Currently supports YouTube videos with English transcripts only.

Built with LangChain, FAISS, Cohere, and OpenAI.

## How it works

1. Fetches the transcript of a YouTube video
2. Splits it into chunks and generates embeddings
3. Stores the embeddings in a FAISS vector store (persisted to disk per video)
4. On each query, retrieves the most relevant chunks using hybrid search (FAISS + BM25)
5. Reranks the retrieved chunks using Cohere Rerank
6. Passes the top chunks to an LLM to generate an answer

## Project Structure
```
youtube-rag/
├── src/
│   ├── indexing.py      # Transcript fetching, chunking, embedding, saving
│   ├── retrieval.py     # Hybrid search (FAISS + BM25) and Cohere reranking
│   └── chain.py         # LCEL chain assembly
├── data/                # Local data files
├── vectorstore/         # Persisted FAISS indexes (one folder per video)
├── notebooks/           # Experimentation notebooks
├── app.py               # Streamlit UI entry point
├── main.py              # CLI entry point for development and testing
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
OPENAI_API_KEY=        # Required for embeddings and LLM
COHERE_API_KEY=        # Required for reranking
```

## Usage

**Run the Streamlit app:**
```bash
streamlit run app.py
```

1. Paste a YouTube URL into the input field
2. Click "Load Video" — the transcript will be fetched, chunked, and indexed
3. Ask questions about the video in the chat interface

If you load the same video again, it loads from the existing index without re-embedding.

**Run from the CLI (for testing):**
```bash
python main.py
```

## Limitations

- Only supports YouTube videos with English transcripts
- Vector store is persisted to disk — not suitable for multi-user deployment as-is
- Switching to a cloud vector database (e.g. Pinecone) is required for production deployment

## Tech Stack

- [LangChain](https://www.langchain.com/) — LLM framework and LCEL chain
- [FAISS](https://github.com/facebookresearch/faiss) — vector store
- [BM25](https://github.com/dorianbrown/rank_bm25) — keyword search
- [Cohere Rerank](https://cohere.com/) — reranking
- [OpenAI](https://openai.com/) — embeddings and LLM
- [Streamlit](https://streamlit.io/) — UI
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — transcript fetching