# Load Test Results

Gathered with [Locust](https://locust.io) (`scripts/locustfile.py`) against a
local backend running in **Qdrant server mode** with rate limiting disabled
(so the script can push past the per-user 20/min quota). The question pool
matches the UI's example pills; the tiny fresh-question pool exercises
retrieval + LLM.

## Run

```
# backend running with RATE_LIMIT_ENABLED=false
python -m locust -f scripts/locustfile.py --host http://localhost:8001 \
    --headless -u 5 -r 1 --run-time 60s --only-summary
```

## Summary (2026-08-25)

| Request      | # reqs | fails | avg (ms) | p50 | p95 | req/s |
|--------------|--------|-------|----------|-----|-----|-------|
| `POST /query`| 515    | 0     | 26       | 13  | 31  | 8.76  |
| `GET /health`| 80     | 0     | 10       | 9   | 22  | 1.36  |
| `POST /auth/login` | 5 | 0 | 2358 | 2316 | —  | 0.09 |

**Aggregate: 600 requests, 0 failures, ~10.2 req/s.**

### Reading

- **`/query` p50 = 13 ms, p95 = 31 ms with 0 failures** — the Redis cache
  serves repeated/scoped questions in tens of milliseconds; the only tail
  spikes (>2 s) are the handful of fresh-question warmup misses that hit the
  LLM at the start of the run.
- **No "cache MISS in steady state" failures** — everything after warmup came
  from Redis, confirming cache keys stay stable across `(scope, question)`.
- **Login stays ~2.4 s** — that is bcrypt work factor 12 (deliberate); it is
  CPU-bound, not a bottleneck signal, and it is rate-limited to 5/min/user in
  production.

### Scale notes

- Single uvicorn worker. Concurrency is unlocked by the per-category Qdrant
  **server** (no single-process file lock) and the Redis cache, so scaling to
  N workers multiplies the cached-path throughput.
- For a "worst case with fresh LLM calls" profile, replace most of
  `CACHED_QUESTIONS` with `FRESH_QUESTIONS` and watch LLM latency/rate limits
  rather than app latency.