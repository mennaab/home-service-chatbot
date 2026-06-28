from fastapi import FastAPI
from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse, ChatData, ErrorResponse
from app.rag.pipeline import ask
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

# =========================================================
# SERVICE TYPE → MongoDB _id MAPPING
# =========================================================
# يتحدث من هنا لما الباكند يضيف خدمة جديدة

SERVICE_TYPE_ID_MAP: dict[str, str] = {
    "Plumbing":               "6a19031f40ac4fdc574cc5da",
    "Electrical":             "6a19031f40ac4fdc574cc5db",
    "Carpentry":              "6a19031f40ac4fdc574cc5dc",
    "Cleaning":               "6a19031f40ac4fdc574cc5dd",
    "Painting":               "6a19031f40ac4fdc574cc5de",
    "AC Technician":          "6a19031f40ac4fdc574cc5df",
    "Internet Technician":    "6a19031f40ac4fdc574cc5e0",
    "Appliance Repair":       "6a19031f40ac4fdc574cc5e1",
    "Handyman":               "6a19031f40ac4fdc574cc5e2",
    "CCTV Installation":      "6a19031f40ac4fdc574cc5e3",
    "Furniture Moving":       "6a19031f40ac4fdc574cc5e4",
    "Gardening":              "6a19031f40ac4fdc574cc5e5",
    "Pest Control":           "6a19031f40ac4fdc574cc5e6",
    "Water Heater Technician":"6a19031f40ac4fdc574cc5e7",
    "Satellite Technician":   "6a19031f40ac4fdc574cc5e8",
    "Locksmith":              "6a19031f40ac4fdc574cc5e9",
    "Gas":                    "6a3ebed187367fb38ec3852e",
}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        history = []
        for msg in request.chat_history:
            if msg.role.value == "user":
                history.append(HumanMessage(content=msg.text))
            else:
                history.append(AIMessage(content=msg.text))

        ai_response = ask(request.message, history)

        # Resolve service_type_id from service_type name
        service_type = ai_response.get("service_type")
        service_type_id = SERVICE_TYPE_ID_MAP.get(service_type) if service_type else None

        return ChatResponse(
            success=True,
            data=ChatData(
                session_id=request.session_id,
                answer=ai_response.get("text_response", ""),
                response_type=ai_response.get("response_type", "rag"),
                service_type=service_type,
                service_type_id=service_type_id,
                issue_description=ai_response.get("issue_description"),
                provider_name=ai_response.get("provider_name"),
                governorate=ai_response.get("governorate"),
                city=ai_response.get("city"),
                street=ai_response.get("street"),
                exact_location=ai_response.get("exact_location"),
                preferred_date=ai_response.get("preferred_date"),
                preferred_time=ai_response.get("preferred_time"),
                payment_mode=ai_response.get("payment_mode"),
                preferred_price=ai_response.get("preferred_price"),
                search_scope=ai_response.get("search_scope"),
            ),
            error=None,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ChatResponse(
            success=False,
            data=None,
            error=ErrorResponse(
                code="INTERNAL_SERVER_ERROR",
                message=str(e),
            ),
        )
