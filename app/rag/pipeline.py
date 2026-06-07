import os

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
# LLM
# =========================================================

llm = ChatGroq(
    model_name=MODEL_NAME
)

# =========================================================
# TEST FUNCTION
# =========================================================

def ask(user_message, chat_history=None):
    response = llm.invoke(user_message)

    return {
        "answer": response.content,
        "chunks_found": 0
    }