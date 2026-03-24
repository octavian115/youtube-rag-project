from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.retrieval import get_retriever


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_chain(video_id: str,  index_name: str = "youtube-rag"):
    retriever = get_retriever(video_id, index_name)

    prompt = PromptTemplate(
    template="""
    You are a helpful assistant that answers questions about a YouTube video based on its transcript.
    
    Use the provided transcript context to answer the question as thoroughly as possible.
    If the context only partially addresses the question, answer based on what is available and mention that the transcript may not cover it fully.
    Only say you don't know if the context contains absolutely no relevant information.

    Context: {context}
    Question: {question}
    
    Answer:
    """,
    input_variables=["context", "question"]
)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    return parallel_chain | prompt | llm | parser