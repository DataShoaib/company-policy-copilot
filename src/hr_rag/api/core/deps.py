from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from hr_rag.api.core.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Stateless JWT auth for production — no DB lookup per request.

    Role and identity come straight from the signed token payload. This avoids a
    database round-trip on every authenticated request, which is the standard
    pattern for stateless microservice auth. Trade-off: if a user is terminated or
    their role changes mid-session, the JWT still carries the old role until it
    expires (short access-token TTL + single-use refresh rotation mitigate this).
    """
    unauthorized = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

    payload = decode_token(token)
    if payload is None:
        raise unauthorized
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh tokens can't authenticate requests — call /auth/refresh first")

    username = payload.get("sub")
    role = payload.get("role")
    if not username or not role:
        raise unauthorized

    return {"username": username, "role": role}
