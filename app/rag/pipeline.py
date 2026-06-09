import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from app.config import *

from app.rag.prompts import (
    contextualize_q_prompt,
    qa_prompt
)

# =========================================================
# ENV
# =========================================================

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# =========================================================
# LOAD DOCUMENTS
# =========================================================

loader = TextLoader(
    "data/knowledge_base.md",
    encoding="utf-8"
)

docs = loader.load()

# =========================================================
# SPLIT DOCUMENTS
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# =========================================================
# VECTOR DATABASE
# =========================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db/chroma"
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model_name=MODEL_NAME
)

# =========================================================
# HISTORY AWARE RETRIEVER
# =========================================================

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# =========================================================
# QUESTION ANSWER CHAIN
# =========================================================

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

# =========================================================
# FINAL RAG CHAIN
# =========================================================

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)

# =========================================================
# MAIN ASK FUNCTION
# =========================================================

def ask(user_message, chat_history):

    docs = retriever.invoke(user_message)

    with open("retrieved_chunks.txt", "w", encoding="utf-8") as f:

        f.write("=" * 50 + "\n")
        f.write("RETRIEVED CHUNKS\n")
        f.write("=" * 50 + "\n\n")

        for i, doc in enumerate(docs):

            f.write(f"CHUNK {i+1}:\n\n")
            f.write(doc.page_content)
            f.write("\n\n" + "-" * 50 + "\n\n")

    response = rag_chain.invoke({
        "input": user_message,
        "chat_history": chat_history
    })

    return response["answer"]