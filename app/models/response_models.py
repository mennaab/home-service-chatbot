from pydantic import BaseModel
from typing import Optional


class ErrorResponse(BaseModel):
    code: str
    message: str


class ChatData(BaseModel):
    answer: str
    session_id: str


class ChatResponse(BaseModel):
    success: bool
    data: Optional[ChatData] = None
    error: Optional[ErrorResponse] = None