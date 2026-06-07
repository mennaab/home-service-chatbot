from fastapi import FastAPI

from app.models.request_models import ChatRequest

from app.models.response_models import (
    ChatResponse,
    ChatData,
    ErrorResponse
)

from app.rag.pipeline import ask

from app.rag.memory import (
    get_history,
    save_message
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

        # get old history
        history = get_history(request.session_id)

        # ask rag
        answer = ask(
            request.message,
            history
        )

        # save user message
        save_message(
            request.session_id,
            "user",
            request.message
        )

        # save assistant message
        save_message(
            request.session_id,
            "assistant",
            answer
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