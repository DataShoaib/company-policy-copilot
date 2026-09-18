"""LLM gateway.

One LiteLLM router owns everything provider related: it picks the model, retries
the call, and falls back to the secondary provider when the primary keeps
failing. Nothing else in the codebase needs to know which provider is in use.
"""

from functools import lru_cache

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel
from litellm import Router

from hr_rag.config import (
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
)

PRIMARY = "hr-primary"
FALLBACK = "hr-fallback"
# prompt caching is not used, so the router's cost table has nothing to record
_NO_CACHE_COST = {"cache_creation_input_token_cost": 0, "cache_read_input_token_cost": 0}


@lru_cache(maxsize=1)
def get_router() -> Router:
    """Groq first, Gemini as fallback, 3 retries before the fallback is used."""
    return Router(
        model_list=[
            {
                "model_name": PRIMARY,
                "litellm_params": {"model": f"groq/{GROQ_MODEL}", "api_key": GROQ_API_KEY},
                "model_info": _NO_CACHE_COST,
            },
            {
                "model_name": FALLBACK,
                "litellm_params": {"model": f"gemini/{GOOGLE_MODEL}", "api_key": GOOGLE_API_KEY},
                "model_info": _NO_CACHE_COST,
            },
        ],
        fallbacks=[{PRIMARY: [FALLBACK]}],
        num_retries=3,
        timeout=600,
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Chat model bound to the router, so every call gets its retries + fallback."""
    llm = ChatLiteLLM(model=PRIMARY, temperature=LLM_TEMPERATURE)
    llm.client = get_router()  # ChatLiteLLM defaults to the bare litellm module
    return llm
