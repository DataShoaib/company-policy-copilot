"""End-to-end API integration tests.

Unlike unit tests, these exercise the *wired routers* over the real ASGI app
(minus the lifespan side-effects: no pipeline warmup, no embedding download,
no Qdrant/Redis). They validate the complete auth journey — signup -> login ->
refresh rotation -> RBAC enforcement — plus the fail-closed / denied query
path, all without network or API keys. The rate limiter is disabled in
``tests/conftest.py``.

The tested query path is deliberately the RBAC-denied + cache-miss one: it
short-circuits before the LLM/Qdrant pipeline is ever touched, so the suite
stays self-contained.
"""
import sys
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hr_rag.api.routes import auth, health, query

app = FastAPI()
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(query.router)


def _new_user() -> dict:
    unique = uuid4().hex[:8]
    return {
        "username": f"e2e-{unique}",
        "password": "StrongPass123",
        "full_name": "E2E Tester",
    }


class TestHealth:
    def test_health_is_reachable_and_pipeline_not_loaded(self):
        # TestClient doesn't run the lifespan, so the pipeline is never warmed
        # — that's the stable, environment-independent assertion. Redis state
        # depends on whether a local/dev Redis is running, so it's not
        # asserted here.
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")
        assert body["pipeline_loaded"] is False


class TestAuthJourney:
    def test_signup_login_refresh_rotation(self):
        client = TestClient(app)
        user = _new_user()

        # 1. signup creates an employee and returns a token pair
        signup = client.post("/auth/signup", json={**user, "role": "employee"})
        assert signup.status_code == 200
        assert signup.json()["role"] == "employee"
        first_refresh = signup.json()["refresh_token"]

        # 2. login with the new credentials works and issues a fresh pair
        login = client.post("/auth/login", json={
            "username": user["username"], "password": user["password"]})
        assert login.status_code == 200
        access = login.json()["access_token"]
        refresh = login.json()["refresh_token"]

        # 3. authorized request: POST /query with a valid access token
        denied = client.post("/query", json={
            "question": "How is the annual bonus calculated?",
            "category": "compensation",  # employee role cannot see this
        }, headers={"Authorization": f"Bearer {access}"})
        assert denied.status_code == 200
        assert denied.json()["sources"] == []
        assert "don't have access" in denied.json()["answer"].lower()

        # 4. refresh rotates the token; the SAME refresh token cannot be reused
        reused_1 = client.post("/auth/refresh", json={"refresh_token": refresh})
        assert reused_1.status_code == 200
        reused_2 = client.post("/auth/refresh", json={"refresh_token": refresh})
        assert reused_2.status_code == 401  # single-use rotation bites

        # 5. the first refresh token (from signup) is also still valid once
        #    — rotation only consumed the login one
        other = client.post("/auth/refresh", json={"refresh_token": first_refresh})
        assert other.status_code == 200

    def test_garbage_credentials_rejected(self):
        client = TestClient(app)
        resp = client.post("/auth/login", json={
            "username": "employee1", "password": "definitely-wrong"})
        assert resp.status_code == 401

    def test_query_requires_auth(self):
        client = TestClient(app)
        resp = client.post("/query", json={"question": "leave policy?"})
        assert resp.status_code in (401, 403)


class TestRBACProvision:
    def test_employee_cannot_provision(self):
        client = TestClient(app)
        from hr_rag.api.core.security import create_access_token

        resp = client.post("/auth/provision", headers={
            "Authorization": f"Bearer {create_access_token(subject='employee1', role='employee')}"
        }, json={**_new_user(), "role": "manager"})
        assert resp.status_code == 403

    def test_hr_admin_can_provision(self):
        client = TestClient(app)
        from hr_rag.api.core.security import create_access_token

        resp = client.post("/auth/provision", headers={
            "Authorization": f"Bearer {create_access_token(subject='hradmin1', role='hr_admin')}"
        }, json={**_new_user(), "role": "manager"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "manager"

    def test_refresh_token_cannot_authenticate_queries(self):
        client = TestClient(app)
        from hr_rag.api.core.security import create_refresh_token

        resp = client.post("/query", json={"question": "x"},
                           headers={"Authorization": f"Bearer {create_refresh_token(subject='employee1', role='employee')}"})
        assert resp.status_code == 401