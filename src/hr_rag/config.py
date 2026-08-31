import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# gemini-2.0-flash is the most widely available free-quota model (Gemini 2.5 is
# also fine if your region/key allows it); gemini-3.6-flash is not a real model name.
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = 0

DEFAULT_TOP_K = 5
RERANK_CANDIDATE_K = 10  # candidate pool per category before merging down to DEFAULT_TOP_K

CATEGORIES = [
    "leave", "compensation", "conduct", "performance", "recruitment",
    "finance", "it", "legal", "operations",
]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "HR_RAG_Experiments"
LANGCHAIN_PROJECT = "HR-RAG-Experiments"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")


def enable_langsmith_tracing():
    # only flip tracing on when a key is actually configured -- enabling it with
    # an empty key makes every LLM call emit client errors / warnings at runtime
    if not LANGCHAIN_API_KEY:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    return True
