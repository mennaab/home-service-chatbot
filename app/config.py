import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL_NAME = "google/gemini-2.5-flash"
# أو
# MODEL_NAME = "qwen/qwen3-32b"
# أو
# MODEL_NAME = "deepseek/deepseek-chat"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50