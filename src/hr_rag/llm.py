from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from hr_rag.config import (
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)


@lru_cache(maxsize=4)
def get_llm(model: str | None = None, temperature: float = LLM_TEMPERATURE) -> BaseChatModel:
    if LLM_PROVIDER == "google":
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY not set — add a Gemini API key to .env")
        return ChatGoogleGenerativeAI(
            model=model or GOOGLE_MODEL,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
        )

    if LLM_PROVIDER != "groq":
        raise RuntimeError("LLM_PROVIDER must be either 'groq' or 'google'")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — add a Groq API key to .env")
    return ChatGroq(model=model or GROQ_MODEL, temperature=temperature, groq_api_key=GROQ_API_KEY)
