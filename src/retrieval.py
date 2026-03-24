# Note: EnsembleRetriever and ContextualCompressionRetriever are in langchain_classic
# due to langchain 1.x restructuring

import os
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


def load_vectorstore(video_id: str, index_name: str = "youtube-rag") -> PineconeVectorStore:
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=video_id
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load vectorstore: {e}")


def get_retriever(video_id: str, index_name: str = "youtube-rag"):
    vector_store = load_vectorstore(video_id, index_name)

    try:
        # Dense retriever
        faiss_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

        # Sparse retriever - fetch docs from Pinecone for BM25
        docs = vector_store.similarity_search("", k=100)
        #fetch up to 100 chunks for BM25 to index locally
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 6

        # Hybrid retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to set up retriever: {e}")

    try:
        reranker = CohereRerank(
            model="rerank-english-v3.0",
            top_n=4
        )
        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble_retriever
        )
    except Exception as e:
        raise RuntimeError(f"Failed to set up reranker: {e}")


# def compare_retrievers(question: str, save_path: str = "vectorstore"):
#     vector_store = load_vectorstore(save_path)

#     # Dense only
#     faiss_retriever = vector_store.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 4}
#     )

#     # Sparse only
#     docs = list(vector_store.docstore._dict.values())
#     bm25_retriever = BM25Retriever.from_documents(docs)
#     bm25_retriever.k = 4

#     faiss_results = faiss_retriever.invoke(question)
#     bm25_results = bm25_retriever.invoke(question)

#     print("\n--- FAISS (dense) results ---")
#     for i, doc in enumerate(faiss_results):
#         print(f"\nChunk {i+1}:\n{doc.page_content[:200]}")

#     print("\n--- BM25 (sparse) results ---")
#     for i, doc in enumerate(bm25_results):
#         print(f"\nChunk {i+1}:\n{doc.page_content[:200]}")