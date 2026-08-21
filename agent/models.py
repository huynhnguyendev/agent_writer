import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

# ============================================================
# 4. LLM
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Please add it to .env"
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please add it to .env"
    )


GEMINI_MAIN_MODEL = os.getenv(
    "GEMINI_MAIN_MODEL",
    "gemini-3.1-flash-lite",
)

GEMINI_FAST_MODEL = os.getenv(
    "GEMINI_FAST_MODEL",
    "gemini-3.5-flash-lite",
)

GROQ_WORKER_MODEL = os.getenv(
    "GROQ_WORKER_MODEL",
    "openai/gpt-oss-120b",
)


# ------------------------------------------------------------
# Gemini = model chính
# ------------------------------------------------------------

gemini_main = ChatGoogleGenerativeAI(
    model=GEMINI_MAIN_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ------------------------------------------------------------
# Gemini Flash-Lite = task nhẹ
# ------------------------------------------------------------

gemini_fast = ChatGoogleGenerativeAI(
    model=GEMINI_FAST_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ------------------------------------------------------------
# Groq = Worker writer
# ------------------------------------------------------------

groq_worker = ChatGroq(
    model=GROQ_WORKER_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)


print("\n========== MODELS ==========")
print("Gemini main :", GEMINI_MAIN_MODEL)
print("Gemini fast :", GEMINI_FAST_MODEL)
print("Groq worker :", GROQ_WORKER_MODEL)