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


# =========================================
# HEALTH ENDPOINT
# =========================================
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================
# CHAT ENDPOINT
# =========================================
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        history = []

        for msg in request.chat_history:

            if msg.role.value == "user":

                history.append(
                    HumanMessage(
                        content=msg.text
                    )
                )

            else:

                history.append(
                    AIMessage(
                        content=msg.text
                    )
                )

        answer = ask(
            request.message,
            history
        )

        return ChatResponse(
            success=True,
            data=ChatData(
                answer=answer,
                session_id=request.session_id
            ),
            error=None
        )

    except Exception as e:

        return ChatResponse(
            success=False,
            data=None,
            error=ErrorResponse(
                code="INTERNAL_SERVER_ERROR",
                message=str(e)
            )
        )