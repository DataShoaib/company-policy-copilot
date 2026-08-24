import sys
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hr_rag.api.core.rbac import VALID_ROLES, allowed_categories_for_role
from hr_rag.api.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from hr_rag.api.core.users import authenticate_user, get_user
from hr_rag.api.routes.auth import router

app = FastAPI()
app.include_router(router)


class TestPasswordHashing:
    def test_hash_and_verify_roundtrip(self):
        hashed = hash_password("correct-horse-battery-staple")
        assert verify_password("correct-horse-battery-staple", hashed)

    def test_wrong_password_rejected(self):
        hashed = hash_password("correct-horse-battery-staple")
        assert not verify_password("wrong-password", hashed)

    def test_hash_is_not_plaintext(self):
        assert hash_password("mypassword") != "mypassword"


class TestJWT:
    def test_access_token_roundtrip(self):
        token = create_access_token(subject="employee1", role="employee")
        payload = decode_token(token)
        assert payload["sub"] == "employee1"
        assert payload["role"] == "employee"
        assert payload["type"] == "access"

    def test_refresh_token_has_refresh_type(self):
        token = create_refresh_token(subject="employee1", role="employee")
        assert decode_token(token)["type"] == "refresh"

    def test_garbage_token_returns_none(self):
        assert decode_token("this.is.not.a.valid.jwt") is None

    def test_tampered_token_returns_none(self):
        token = create_access_token(subject="employee1", role="employee")
        assert decode_token(token[:-4] + "abcd") is None


class TestUserAuth:
    def test_correct_credentials_succeed(self):
        user = authenticate_user("employee1", "employee123")
        assert user is not None
        assert user["role"] == "employee"

    def test_wrong_password_fails(self):
        assert authenticate_user("employee1", "wrong") is None

    def test_unknown_username_fails(self):
        assert authenticate_user("nobody", "whatever") is None

    def test_get_user_hides_nothing_extra(self):
        assert get_user("hradmin1")["role"] == "hr_admin"


class TestRBAC:
    def test_employee_cannot_see_compensation(self):
        assert "compensation" not in allowed_categories_for_role("employee")

    def test_hr_admin_sees_everything(self):
        allowed = allowed_categories_for_role("hr_admin")
        for cat in ["leave", "compensation", "conduct", "performance", "recruitment"]:
            assert cat in allowed

    def test_manager_sees_performance_but_not_compensation(self):
        allowed = allowed_categories_for_role("manager")
        assert "performance" in allowed
        assert "compensation" not in allowed

    def test_employee_can_search_it_but_not_legal(self):
        allowed = allowed_categories_for_role("employee")
        assert "it" in allowed
        assert "legal" not in allowed

    def test_unknown_role_gets_nothing(self):
        assert allowed_categories_for_role("intern_without_a_defined_role") == []

    def test_all_demo_users_have_valid_roles(self):
        for username in ["employee1", "manager1", "hradmin1"]:
            assert get_user(username)["role"] in VALID_ROLES


class TestSignup:
    def test_signup_creates_user_and_returns_tokens(self):
        client = TestClient(app)
        username = f"newemployee-{uuid4().hex[:8]}"
        payload = {
            "username": username,
            "password": "StrongPass123",
            "full_name": "New Employee",
            "role": "employee",
        }

        response = client.post("/auth/signup", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "employee"
        assert "access_token" in data
        assert "refresh_token" in data
        assert authenticate_user(username, "StrongPass123") is not None

    def test_signup_cannot_self_assign_privileged_role(self):
        client = TestClient(app)
        username = f"untrustedadmin-{uuid4().hex[:8]}"
        response = client.post("/auth/signup", json={
            "username": username,
            "password": "StrongPass123",
            "full_name": "Untrusted User",
            "role": "hr_admin",
        })

        assert response.status_code == 200
        assert response.json()["role"] == "employee"

    def test_non_admin_cannot_provision_user(self):
        client = TestClient(app)
        username = f"newmanager-{uuid4().hex[:8]}"

        response = client.post("/auth/provision", headers={
            "Authorization": f"Bearer {create_access_token(subject='employee1', role='employee')}"
        }, json={
            "username": username,
            "password": "StrongPass123",
            "full_name": "New Manager",
            "role": "manager",
        })

        assert response.status_code == 403

    def test_hr_admin_can_provision_manager(self):
        client = TestClient(app)
        username = f"newmanager-{uuid4().hex[:8]}"

        response = client.post("/auth/provision", headers={
            "Authorization": f"Bearer {create_access_token(subject='hradmin1', role='hr_admin')}"
        }, json={
            "username": username,
            "password": "StrongPass123",
            "full_name": "New Manager",
            "role": "manager",
        })

        assert response.status_code == 200
        assert response.json()["role"] == "manager"
        assert authenticate_user(username, "StrongPass123")["role"] == "manager"
