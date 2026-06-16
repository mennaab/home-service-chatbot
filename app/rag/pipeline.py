import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser

from app.config import GROQ_API_KEY, MODEL_NAME
from app.rag.prompts import contextualize_q_prompt, qa_prompt

# =========================================================
# ENV
# =========================================================

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# =========================================================
# LOAD & SPLIT DOCUMENTS
# =========================================================

loader = TextLoader("data/knowledge_base.md", encoding="utf-8")
docs = loader.load()

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

CHROMA_PATH = "db/chroma"

if os.path.exists(CHROMA_PATH):
    print("Loading existing Chroma DB...")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
else:
    print("Creating Chroma DB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# =========================================================
# LLM
# =========================================================

llm = ChatGroq(model_name=MODEL_NAME)

# =========================================================
# CHAINS
# =========================================================

rephrase_chain = (
    contextualize_q_prompt
    | llm
    | StrOutputParser()
)

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

# =========================================================
# LANGUAGE DETECTOR
# =========================================================

def detect_language(text: str) -> str:
    for char in text:
        if "\u0600" <= char <= "\u06FF":
            return "Arabic"
    return "English"

# =========================================================
# MAIN ASK FUNCTION
# =========================================================

def ask(user_message: str, chat_history: list) -> str:

    # STEP 1: DETECT LANGUAGE
    language = detect_language(user_message)

    # STEP 2: REWRITE QUESTION IF NEEDED
    if chat_history:
        new_question = rephrase_chain.invoke({
            "input": user_message,
            "chat_history": chat_history
        })
    else:
        new_question = user_message

    # STEP 3: RETRIEVE CHUNKS
    retrieved_docs = retriever.invoke(new_question)

    # STEP 4: DEBUG
    with open("retrieved_chunks.txt", "w", encoding="utf-8") as f:
        f.write("DETECTED LANGUAGE: " + language + "\n\n")
        f.write("ORIGINAL QUESTION:\n")
        f.write(user_message)
        f.write("\n\n")
        f.write("REWRITTEN QUESTION:\n")
        f.write(new_question)
        f.write("\n\n")
        f.write("=" * 50 + "\n")
        f.write("RETRIEVED CHUNKS\n")
        f.write("=" * 50 + "\n\n")
        for i, doc in enumerate(retrieved_docs):
            f.write("CHUNK " + str(i+1) + ":\n\n")
            f.write(doc.page_content)
            f.write("\n\n")
            f.write("-" * 50 + "\n\n")
      # STEP 5: GET ANSWER
    response = question_answer_chain.invoke({
        "input": new_question,
        "chat_history": chat_history,
        "context": retrieved_docs,
        "language": language
    })

    # STEP 6: EXTRACT STRING
    if hasattr(response, "content"):
        answer = response.content
    elif isinstance(response, str):
        answer = response
    else:
        answer = str(response)

    return answer