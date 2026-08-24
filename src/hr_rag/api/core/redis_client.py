from functools import lru_cache

import redis

from hr_rag.api.core.settings import settings


@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)
