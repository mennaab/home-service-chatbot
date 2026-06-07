from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

# =========================================================
# CONTEXTUALIZE QUESTION PROMPT
# =========================================================

contextualize_q_prompt = ChatPromptTemplate.from_messages([

    MessagesPlaceholder("chat_history"),

    (
        "human",
        "{input}"
    ),

    (
        "human",
        """
حوّل السؤال إلى سؤال واضح ومفهوم بدون الاعتماد على الشات السابق.
لا تجاوب على السؤال.
"""
    )
])

# =========================================================
# QA PROMPT
# =========================================================

qa_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
أنت مساعد لخدمة عملاء.

اعتمد فقط على المعلومات الموجودة داخل الـ context.

مهم جداً:
- إذا كانت الخدمة غير متاحة اذكر بوضوح أنها غير متاحة.
- ممنوع اختراع خدمات غير موجودة.
- لا تفترض أي معلومة غير موجودة.
- إذا لم تجد إجابة واضحة قل:
"لا أملك معلومات كافية."

أجب بشكل مختصر وواضح.
"""
    ),

    MessagesPlaceholder("chat_history"),

    (
        "human",
        """
السؤال:
{input}

المعلومات المتاحة:
{context}
"""
    )
])