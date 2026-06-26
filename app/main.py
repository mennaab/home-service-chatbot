from fastapi import FastAPI
from app.models.request_models import ChatRequest
from app.models.response_models import (
    ChatResponse,
    ChatData,
    ErrorResponse
)
from app.rag.pipeline import ask
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        history = []
        for msg in request.chat_history:
            if msg.role.value == "user":
                history.append(HumanMessage(content=msg.text))
            else:
                history.append(AIMessage(content=msg.text))

        ai_response = ask(
            request.message,
            history
        )

        return ChatResponse(
            success=True,
            data=ChatData(
                session_id=request.session_id,
                answer=ai_response.get("text_response", ""),
                response_type=ai_response.get("response_type", "rag"),
                service_type=ai_response.get("service_type"),
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
                search_scope=ai_response.get("search_scope")
            ),
            error=None
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ChatResponse(
            success=False,
            data=None,
            error=ErrorResponse(
                code="INTERNAL_SERVER_ERROR",
                message=str(e)
            )
        )