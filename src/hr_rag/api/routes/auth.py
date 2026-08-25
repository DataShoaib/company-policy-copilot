from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from hr_rag.api.core.deps import get_current_user
from hr_rag.api.core.metrics import AUTH_TOTAL
from hr_rag.api.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
from hr_rag.api.core.settings import settings
from hr_rag.api.core.users import (
    authenticate_user,
    consume_refresh_token,
    get_user,
    provision_user,
    register_user,
    store_refresh_token,
)
from hr_rag.api.schemas.schemas import (
    LoginRequest,
    ProvisionUserRequest,
    RefreshRequest,
    SignupRequest,
    TokenResponse,
)
from hr_rag.api.services.rate_limit import (
    RateLimitExceeded,
    RateLimitServiceUnavailable,
    check_login_rate_limit,
    check_rate_limit,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue_tokens(user: dict) -> TokenResponse:
    """Issue an access + refresh pair and persist the refresh token so that
    /auth/refresh can enforce single-use rotation."""
    refresh_token = create_refresh_token(subject=user["username"], role=user["role"])
    store_refresh_token(
        token=refresh_token,
        username=user["username"],
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days),
    )
    return TokenResponse(
        access_token=create_access_token(subject=user["username"], role=user["role"]),
        refresh_token=refresh_token,
        role=user["role"],
    )


@router.post("/signup", response_model=TokenResponse)
def signup(body: SignupRequest):
    try:
        check_rate_limit(body.username.strip().lower(), settings.login_rate_limit_per_minute, namespace="signup")
    except RateLimitExceeded as exc:
        AUTH_TOTAL.labels(kind="signup", outcome="rate_limited").inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many signup attempts. Try again in {exc.retry_after_seconds} seconds.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    except RateLimitServiceUnavailable as exc:
        AUTH_TOTAL.labels(kind="signup", outcome="service_unavailable").inc()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Signup service is temporarily unavailable. Please try again later.") from exc

    try:
        user = register_user(body.username, body.password, body.full_name, "employee")
    except ValueError as exc:
        AUTH_TOTAL.labels(kind="signup", outcome="failure").inc()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    AUTH_TOTAL.labels(kind="signup", outcome="success").inc()
    return _issue_tokens(user)


@router.post("/provision", response_model=TokenResponse)
def provision(body: ProvisionUserRequest, current_user: dict = Depends(get_current_user)):  # noqa: B008
    if current_user["role"] != "hr_admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only HR admins can provision users")

    try:
        user = provision_user(body.username, body.password, body.full_name, body.role)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _issue_tokens(user)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    try:
        check_login_rate_limit(body.username)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many login attempts. Try again in {exc.retry_after_seconds} seconds.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    except RateLimitServiceUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Login service is temporarily unavailable. Please try again later.") from exc

    user = authenticate_user(body.username.strip(), body.password)
    if not user:
        AUTH_TOTAL.labels(kind="login", outcome="failure").inc()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    AUTH_TOTAL.labels(kind="login", outcome="success").inc()

    return _issue_tokens(user)


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest):
    payload = decode_token(body.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")

    username = payload.get("sub")
    # Throttle refresh attempts too — an attacker brute-forcing refresh tokens
    # or replaying a stolen one should not get unbounded guesses.
    try:
        check_rate_limit(username, settings.login_rate_limit_per_minute, namespace="refresh")
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many refresh attempts. Try again in {exc.retry_after_seconds} seconds.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    except RateLimitServiceUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Refresh service is temporarily unavailable. Please try again later.") from exc

    # Single-use rotation: if this token has already been used, or was revoked,
    # reject it. This is what stops a stolen refresh token from being replayed.
    if not consume_refresh_token(body.refresh_token, username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has already been used. Please log in again.")

    user = get_user(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")

    return _issue_tokens(user)