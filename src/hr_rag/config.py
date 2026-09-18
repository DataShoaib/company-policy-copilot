import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
LLM_TEMPERATURE = 0

DEFAULT_TOP_K = 3
RERANK_CANDIDATE_K = 10

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
