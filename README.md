# YouTube RAG Pipeline

A retrieval-augmented generation (RAG) pipeline that answers questions about a YouTube video using its transcript.

Built with LangChain, FAISS, and OpenAI.

## How it works

1. Fetches the transcript of a YouTube video
2. Splits it into chunks and generates embeddings
3. Stores the embeddings in a FAISS vector store (persisted to disk)
4. On each query, retrieves the most relevant chunks and passes them to an LLM to generate an answer

## Project Structure
```
youtube-rag/
├── src/
│   ├── indexing.py      # Transcript fetching, chunking, embedding, saving
│   ├── retrieval.py     # Loading vectorstore and configuring retriever
│   └── chain.py         # LCEL chain assembly
├── data/                # Local data files
├── vectorstore/         # Persisted FAISS index
├── notebooks/           # Experimentation notebooks
├── main.py              # Entry point
└── pyproject.toml       # Dependencies
```

## Setup

1. Clone the repo
2. Create a virtual environment and install dependencies
```bash
uv venv && source .venv/bin/activate
uv sync
```
3. Copy `.env.example` to `.env` and add your OpenAI API key
```bash
cp .env.example .env
```

## Usage

**Index a video (run once):**

In `main.py`, set your `VIDEO_ID` and uncomment `index()`, then run:
```bash
python main.py
```

After indexing, comment out `index()` to avoid re-embedding on every run.

**Ask questions:**
```bash
python main.py
```

## Tech Stack

- [LangChain](https://www.langchain.com/) — LLM framework and LCEL chain
- [FAISS](https://github.com/facebookresearch/faiss) — vector store
- [OpenAI](https://openai.com/) — embeddings and LLM
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — transcript fetching