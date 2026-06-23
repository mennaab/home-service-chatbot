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

# ضفنا الكلاس ده عشان الـ main.py يعرف يعمل له import وميضربش
class ErrorResponse(BaseModel):
    code: str
    message: str

class ChatData(BaseModel):
    answer: str
    session_id: str
    response_type: Optional[ResponseType] = None
    service_type: Optional[ServiceType] = None
    provider_name: Optional[str] = None
    governorate: Optional[Governorate] = None
    city: Optional[City] = None
    street: Optional[str] = None            # الحقل الجديد للشارع
    exact_location: Optional[str] = None    # الحقل الجديد للوكيشن التفصيلي
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None    # الحقل الجديد للوقت
    payment_mode: Optional[PaymentMode] = None      # الحقل الجديد لطريقة الدفع
    preferred_price: Optional[float] = None # الحقل الجديد للسعر
    search_scope: Optional[SearchScope] = None      # الحقل الجديد لنطاق البحث (Governorate / District)

class ChatResponse(BaseModel):
    success: bool
    data: Optional[ChatData] = None
    error: Optional[ErrorResponse] = None