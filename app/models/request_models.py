from pydantic import BaseModel
from typing import List
from datetime import datetime
from enum import Enum


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    text: str
    role: Role
    timestamp: datetime


class ChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: List[ChatMessage]