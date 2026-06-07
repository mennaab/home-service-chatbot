from app.rag.pipeline import ask_rag

from app.rag.memory import (
    get_chat_history,
    save_message
)


def chat_service(request):

    history = get_chat_history(
        request.session_id
    )

    answer = ask_rag(

        session_id=request.session_id,

        user_message=request.message,

        chat_history=history
    )

    save_message(
        request.session_id,
        "user",
        request.message
    )

    save_message(
        request.session_id,
        "assistant",
        answer
    )

    return {

        "success": True,

        "data": {

            "session_id": request.session_id,

            "answer": answer
        },

        "error": None
    }