import hashlib
import json

import redis

from hr_rag.api.core.redis_client import get_redis
from hr_rag.api.core.settings import settings

# v4: cache key now includes the requested category scope. Previously the key
# was derived from the question text alone, so a category-scoped query ("leave")
# could overwrite a good unscoped answer for the same question with a scoped
# miss ("I don't have information"), which was then served to everyone.
CACHE_PREFIX = "hrrag:answer:v4:"


def _cache_key(question: str, category: str | None = None) -> str:
    scope = category.strip().lower() if category else "__all__"
    digest = hashlib.sha256(f"{scope}|{question.strip().lower()}".encode()).hexdigest()
    return f"{CACHE_PREFIX}{digest}"


def get_cached_answer(question: str, category: str | None = None) -> dict | None:
    try:
        raw = get_redis().get(_cache_key(question, category))
    except redis.exceptions.RedisError:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        # corrupted entry -- drop it rather than crashing every request on this key
        try:
            get_redis().delete(_cache_key(question, category))
        except redis.exceptions.RedisError:
            pass
        return None


def set_cached_answer(question: str, answer: str, sources: list, category: str | None = None) -> None:
    payload = json.dumps({"answer": answer, "sources": sources})
    try:
        get_redis().setex(_cache_key(question, category), settings.cache_ttl_seconds, payload)
    except redis.exceptions.RedisError:
        pass


def cached_answer_is_allowed(cached: dict, allowed_categories: list, category: str | None = None) -> bool:
    source_categories = {source.get("category") for source in cached.get("sources", [])}
    # an empty-source cached answer is a "couldn't find anything" fallback --
    # never reuse it, a user with broader access may be entitled to a real answer
    if not source_categories:
        return False
    if category and category not in allowed_categories:
        return False
    if category and source_categories != {category}:
        return False
    return source_categories.issubset(set(allowed_categories))