import hashlib
import json
from typing import Any

import redis

from hr_rag.api.core.redis_client import get_redis
from hr_rag.api.core.settings import settings

CACHE_PREFIX = "hrrag:answer:v5:"
ALL_CATEGORIES = "__all__"


def scope_fingerprint(allowed_categories: list[str]) -> str:
    """Stable tag for a role's category scope, so one role never reads another's cached answer."""
    joined = ",".join(sorted(allowed_categories))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _cache_key(question: str, scope: str, category: str | None = None) -> str:
    category_scope = category.strip().lower() if category else ALL_CATEGORIES
    normalized_question = question.strip().lower()
    cache_key_input = f"{scope}|{category_scope}|{normalized_question}"
    question_hash = hashlib.sha256(cache_key_input.encode("utf-8")).hexdigest()
    return f"{CACHE_PREFIX}{question_hash}"


def get_cached_answer(question: str, scope: str, category: str | None = None) -> dict[str, Any] | None:
    redis_client = get_redis()
    cache_key = _cache_key(question, scope, category)
    try:
        cached_data = redis_client.get(cache_key)
    except redis.exceptions.RedisError:
        return None
    if not cached_data:
        return None
    try:
        return json.loads(cached_data)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            redis_client.delete(cache_key)
        except redis.exceptions.RedisError:
            pass
        return None


def set_cached_answer(question: str, answer: str, sources: list, scope: str, category: str | None = None) -> None:
    cache_payload = json.dumps({"answer": answer, "sources": sources})
    cache_key = _cache_key(question, scope, category)
    try:
        get_redis().setex(cache_key, settings.cache_ttl_seconds, cache_payload)
    except redis.exceptions.RedisError:
        pass


def cached_answer_is_allowed(cached: dict, allowed_categories: list, category: str | None = None) -> bool:
    if category and category not in allowed_categories:
        return False

    cached_sources = cached.get("sources", [])
    if not isinstance(cached_sources, list):
        return False

    cached_source_categories = {source.get("category") for source in cached_sources if isinstance(source, dict)}
    if not cached_source_categories:
        return False

    if category and cached_source_categories != {category}:
        return False

    return cached_source_categories.issubset(set(allowed_categories))