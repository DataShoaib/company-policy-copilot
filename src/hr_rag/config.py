import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Anchored to the repo instead of the current directory, so the API, the
# scripts and the notebook all read the same file from any working directory.
load_dotenv(PROJECT_ROOT / ".env")

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

LANGCHAIN_PROJECT = os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "HR-RAG-Experiments"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGSMITH_API_KEY", "") or os.getenv("LANGCHAIN_API_KEY", "")


def enable_langsmith_tracing(project: str | None = None) -> bool:
    """Turn on LangSmith tracing. Returns False when no API key is configured.

    Enabling tracing with an empty key makes every LLM call emit client errors,
    so tracing stays off unless a key is present. Both the LANGSMITH_* and the
    legacy LANGCHAIN_* variable names are set because the SDK checks the
    LANGSMITH_* namespace first.
    """
    if not LANGCHAIN_API_KEY:
        return False

    project = project or LANGCHAIN_PROJECT
    for namespace in ("LANGSMITH", "LANGCHAIN"):
        os.environ[f"{namespace}_API_KEY"] = LANGCHAIN_API_KEY
        os.environ[f"{namespace}_PROJECT"] = project
        os.environ[f"{namespace}_TRACING"] = "true"
        os.environ[f"{namespace}_TRACING_V2"] = "true"

    # langsmith memoises both the raw env lookups (get_env_var) and the resolved
    # project name (get_tracer_project). If either was read before this point the
    # cached default wins and traces land in a project called "default", so both
    # caches have to be dropped after the variables above are set.
    from langsmith import utils as langsmith_utils

    langsmith_utils.get_env_var.cache_clear()
    langsmith_utils.get_tracer_project.cache_clear()
    return True
