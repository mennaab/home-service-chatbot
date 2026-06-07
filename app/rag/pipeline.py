import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from app.config import *

# =========================================================
# ENV
# =========================================================

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# =========================================================
# LOAD EXISTING CHROMA ONLY
# =========================================================

vector_store = Chroma(
    persist_directory="db/chroma",
    embedding_function=embeddings
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
# TEST
# =========================================================

def ask(user_message, chat_history):
    docs = retriever.invoke(user_message)

    return {
        "chunks_found": len(docs)
    }