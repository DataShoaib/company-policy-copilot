import os
from functools import lru_cache

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from hr_rag.api.core.metrics import LLM_CALLS, LLM_FALLBACKS
from hr_rag.config import (
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)

# LiteLLM reads Gemini keys from GEMINI_API_KEY; our .env uses GOOGLE_API_KEY.
if GOOGLE_API_KEY:
    os.environ.setdefault("GEMINI_API_KEY", GOOGLE_API_KEY)


@lru_cache(maxsize=8)
def get_llm(provider: str | None = None, model: str | None = None, temperature: float = LLM_TEMPERATURE) -> BaseChatModel:
    provider = (provider or LLM_PROVIDER).lower()
    if provider == "google":
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY not set — add a Gemini API key to .env")
        return ChatLiteLLM(
            model_name=f"gemini/{model or GOOGLE_MODEL}",
            temperature=temperature,
            model_kwargs={"num_retries": 3, "timeout": 600},
        )
    if provider != "groq":
        raise RuntimeError("LLM_PROVIDER must be either 'groq' or 'google'")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — add a Groq API key to .env")
    return ChatLiteLLM(
        model_name=f"groq/{model or GROQ_MODEL}",
        temperature=temperature,
        model_kwargs={"num_retries": 3, "timeout": 600},
    )


def _secondary_provider() -> str:
    return "google" if LLM_PROVIDER == "groq" else "groq"


def _provider_configured(provider: str) -> bool:
    return bool(GOOGLE_API_KEY if provider == "google" else GROQ_API_KEY)


def build_answer_llm() -> Runnable:
    """A runnable that answers with the configured provider and, on failure,
    transparently falls back to the other provider.

    LangChain's ``with_fallbacks`` runs the secondary runnable whenever the
    primary raises, so a Groq outage no longer hard-fails the API — Gemini is
    the safety net (and vice-versa). Every call is counted for /metrics.
    """
    primary = LLM_PROVIDER
    secondary = _secondary_provider()

    def _run(inp, provider: str, fallback: bool = False):
        llm = get_llm(provider)
        if fallback:
            LLM_FALLBACKS.inc()
            LLM_CALLS.labels(provider=provider, result="fallback_ok").inc()
        else:
            LLM_CALLS.labels(provider=provider, result="primary_ok").inc()
        return llm.invoke(inp)

    main_runner: Runnable = RunnableLambda(lambda inp: _run(inp, primary))
    if _provider_configured(secondary):
        fallback_runner: Runnable = RunnableLambda(lambda inp: _run(inp, secondary, fallback=True))
        return main_runner.with_fallbacks([fallback_runner])
    return main_runner
