from pydantic import BaseModel
from typing import Optional

from app.models.enums import (
    ResponseType,
    ServiceType,
    Governorate,
    City,
    PaymentMode,
    SearchScope,
)

class ErrorResponse(BaseModel):
    code: str
    message: str

class ChatData(BaseModel):
    answer: str
    session_id: str
    response_type: Optional[ResponseType] = None
    service_type: Optional[ServiceType] = None
    service_type_id: Optional[str] = None        # MongoDB _id for service
    issue_description: Optional[str] = None
    provider_name: Optional[str] = None
    governorate: Optional[Governorate] = None
    city: Optional[City] = None
    street: Optional[str] = None
    exact_location: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    payment_mode: Optional[PaymentMode] = None
    preferred_price: Optional[float] = None
    search_scope: Optional[SearchScope] = None

class ChatResponse(BaseModel):
    success: bool
    data: Optional[ChatData] = None
    error: Optional[ErrorResponse] = None
