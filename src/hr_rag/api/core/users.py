import hashlib
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from hr_rag.api.core.database import Base, SessionLocal
from hr_rag.api.core.rbac import VALID_ROLES
from hr_rag.api.core.security import hash_password, verify_password


class User(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(100), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)


class RefreshToken(Base):
    """One row per issued JWT refresh token. Rows are single-use: consuming one
    during rotation invalidates it, so a stolen refresh token cannot be handed
    out / replayed for the token's full lifetime."""

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(), nullable=False)  # naive UTC


def _as_dict(user: User) -> dict:
    return {
        "username": user.username,
        "full_name": user.full_name,
        "hashed_password": user.hashed_password,
        "role": user.role,
    }


def get_user(username: str) -> dict | None:
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == username))
        return _as_dict(user) if user else None


def register_user(username: str, password: str, full_name: str, role: str) -> dict:
    normalized_username = username.strip()
    if not normalized_username:
        raise ValueError("Username is required")
    if not full_name.strip():
        raise ValueError("Full name is required")
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role. Allowed roles: {', '.join(VALID_ROLES)}")
    user = User(
        username=normalized_username,
        full_name=full_name.strip(),
        hashed_password=hash_password(password),
        role=role,
    )
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.username == normalized_username)) is not None:
            raise ValueError("Username already exists")
        db.add(user)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise ValueError("Username already exists") from exc
        db.refresh(user)
        return _as_dict(user)


def provision_user(username: str, password: str, full_name: str, role: str) -> dict:
    return register_user(username, password, full_name, role)


def authenticate_user(username: str, password: str) -> dict | None:
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def seed_demo_users() -> None:
    demo_users = [
        ("employee1", "employee123", "Aisha Khan", "employee"),
        ("manager1", "manager123", "Rahul Verma", "manager"),
        ("hradmin1", "hradmin123", "Priya Nair", "hr_admin"),
    ]
    for username, password, full_name, role in demo_users:
        if get_user(username) is None:
            register_user(username, password, full_name, role)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def store_refresh_token(token: str, username: str, expires_at: datetime) -> None:
    # Store a naive UTC timestamp so the expiry check below is safe on both
    # SQLite (which returns naive datetimes) and PostgreSQL.
    if expires_at.tzinfo is not None:
        expires_at = expires_at.astimezone(timezone.utc).replace(tzinfo=None)
    with SessionLocal() as db:
        db.add(RefreshToken(username=username, token_hash=_token_hash(token), expires_at=expires_at))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()


def consume_refresh_token(token: str, username: str) -> bool:
    """Single-use: delete the stored row for this token. Once consumed it can
    never be presented again, which is what makes refresh rotation work."""
    token_hash = _token_hash(token)
    with SessionLocal() as db:
        row = db.scalar(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.username == username,
            )
        )
        if row is None:
            return False
        db.delete(row)
        db.commit()
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        return row.expires_at is None or row.expires_at >= utc_now


def revoke_user_refresh_tokens(username: str) -> None:
    with SessionLocal() as db:
        db.execute(delete(RefreshToken).where(RefreshToken.username == username))
        db.commit()