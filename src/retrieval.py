# Note: EnsembleRetriever and ContextualCompressionRetriever are in langchain_classic
# due to langchain 1.x restructuring

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


def load_vectorstore(save_path: str = "vectorstore") -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_retriever(save_path: str = "vectorstore"):
    vector_store = load_vectorstore(save_path)

    # Dense retriever
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # Sparse retriever
    docs = list(vector_store.docstore._dict.values())
    #this pulls the document chunks out of FAISS so that BM25 can index them
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 6

    # Combine both
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # Reranker
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=3
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )


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