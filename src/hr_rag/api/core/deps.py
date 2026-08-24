from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from hr_rag.api.core.security import decode_token
from hr_rag.api.core.users import get_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    unauthorized = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

    payload = decode_token(token)
    if payload is None:
        raise unauthorized
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh tokens can't authenticate requests — call /auth/refresh first")

    user = get_user(payload.get("sub"))
    if user is None:
        raise unauthorized

    return {"username": user["username"], "full_name": user["full_name"], "role": user["role"]}
