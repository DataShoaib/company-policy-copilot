"""Locust load-test for the Policy Copilot API.

Usage (backend must be running; run with rate limiting disabled so the test
can push concurrency past the per-user 20/min quota):

    $env:RATE_LIMIT_ENABLED = "false"  # PowerShell
    python -m locust -f scripts/locustfile.py --host http://localhost:8001 \
        --headless -u 8 -r 2 --run-time 120s

Mix: the bulk of the load re-asks a small pool of questions so it exercises
the Redis cache path (cheap, no LLM spend); a minority uses fresh questions
that hit retrieval + the LLM.
"""
from __future__ import annotations

from locust import HttpUser, between, events, task

# Realistic question pool the UI ships as examples. Repeating these measures
# throughput of the cached path plus auth + rate-limit overhead.
CACHED_QUESTIONS = [
    "How many days of casual leave do I get per year?",
    "What is the employer PF contribution?",
    "Can I carry forward unused casual leave?",
    "What are the password rules?",
    "What do I do after clicking a phishing link?",
    "How is the annual bonus calculated?",
]

FRESH_QUESTIONS = [
    "What is the notice period for resignation after probation?",
    "How many working days is paternity leave for a new father?",
    "What is the travel reimbursement limit for domestic flights?",
    "Describe the grievance redressal process in two sentences.",
    "What documentation is needed for a referral bonus payout?",
]


@events.test_start.add_listener
def _on_test_start(environment, **_kwargs):
    """Warm the cache with the question pool so most load hits Redis."""
    import requests

    try:
        login = requests.post(
            f"{environment.host}/auth/login",
            json={"username": "employee1", "password": "employee123"},
            timeout=15,
        )
        token = login.json()["access_token"]
        for q in CACHED_QUESTIONS:
            requests.post(
                f"{environment.host}/query",
                json={"question": q},
                headers={"Authorization": f"Bearer {token}"},
                timeout=90,
            )
        print("[warmup] cache primed with", len(CACHED_QUESTIONS), "questions")
    except Exception as exc:  # noqa: BLE001 - warmup is best-effort
        print("[warmup] skipped:", exc)


class PolicyUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._q = 0

    def on_start(self):
        resp = self.client.post(
            "/auth/login",
            json={"username": "employee1", "password": "employee123"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"login failed (HTTP {resp.status_code}) during load test")
        self.token = resp.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def _next(self, pool: list[str]) -> str:
        question = pool[self._q % len(pool)]
        self._q += 1
        return question

    @task(7)
    def cached_query(self):
        q = self._next(CACHED_QUESTIONS)
        with self.client.post(
            "/query",
            json={"question": q},
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
            elif not resp.json().get("cached"):
                # A warmup miss is fine; a steady-state miss means TTL expiry
                # or the cache key diverged — worth surfacing.
                resp.failure("cache MISS in steady state")

    @task(1)
    def fresh_query(self):
        q = self._next(FRESH_QUESTIONS)
        with self.client.post(
            "/query",
            json={"question": q},
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health")