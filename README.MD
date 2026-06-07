# RAG Chatbot API

AI chatbot API built using FastAPI + LangChain + RAG architecture.

The chatbot answers user questions using a custom knowledge base with conversation history support.

---

# Features

- RAG (Retrieval Augmented Generation)
- Chat History Support
- FastAPI REST API
- Swagger Documentation
- Multilingual Embeddings
- Arabic + English Support
- Validation & Error Handling
- HuggingFace Deployment

---

# Tech Stack

- FastAPI
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq LLM
- Python

---

# API Endpoints

## Health Check

GET /health
Response:

{
  "status": "healthy"
}
---

## Chat Endpoint

POST /chat
Request Body:

{
  "session_id": "123",
  "message": "عندي مشكلة سباكة"
}
Success Response:

{
  "success": true,
  "data": {
    "answer": "يمكنك طلب خدمة سباكة...",
    "session_id": "123"
  },
  "error": null
}
Error Response:

{
  "success": false,
  "data": null,
  "error": {
    "code": "INTERNAL_SERVER_ERROR",
    "message": "Error message"
  }
}
---

# Project Structure

project/
│
├── app/
├── data/
├── db/
├── requirements.txt
├── Dockerfile
└── README.md
---

# Run Locally

uvicorn app.main:app --reload
---

# Swagger Docs

/docs
---

# Deployment

Deployed using HuggingFace Spaces.