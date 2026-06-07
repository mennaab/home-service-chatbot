from fastapi import APIRouter

from app.models.request_models import ChatRequest
from app.services.rag_service import chat_service

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):

    response = chat_service(request)

    return response