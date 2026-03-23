from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.retrieval import get_retriever


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_chain(save_path: str = "vectorstore"):
    retriever = get_retriever(save_path)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context: {context}
        Question: {question}
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