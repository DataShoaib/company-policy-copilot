import time

import redis

from hr_rag.api.core.redis_client import get_redis
from hr_rag.api.core.settings import settings


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: int):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"rate limit exceeded, retry after {retry_after_seconds}s")


class RateLimitServiceUnavailable(Exception):
    """Redis is unavailable, so requests must not bypass rate limiting."""


def check_rate_limit(user_id: str, limit_per_minute: int | None = None, namespace: str = "query") -> None:
    if not settings.rate_limit_enabled:
        return
    limit = limit_per_minute or settings.rate_limit_per_minute
    window = int(time.time() // 60)
    key = f"hrrag:rate:{namespace}:{user_id}:{window}"

    try:
        r = get_redis()
        count = r.incr(key)
        if count == 1:
            r.expire(key, 60)
    except redis.exceptions.RedisError as exc:
        raise RateLimitServiceUnavailable("Redis is required for rate limiting") from exc

    if count > limit:
        raise RateLimitExceeded(retry_after_seconds=60 - (int(time.time()) % 60))


def check_login_rate_limit(username: str) -> None:
    check_rate_limit(username.strip().lower(), settings.login_rate_limit_per_minute, namespace="login")
