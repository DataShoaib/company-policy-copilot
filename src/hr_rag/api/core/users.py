from datetime import datetime, timezone

from sqlalchemy import BigInteger, String, delete, select
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


class ConsumedRefreshToken(Base):
    """Ledger of already-redeemed refresh-token JTIs (single-use rotation).

    Lives in the application DB so rotation is enforced even without Redis.
    Rows are only needed until their token's own expiry and are purged
    opportunistically.
    """

    __tablename__ = "consumed_refresh_tokens"

    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    expires_at: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)


def consume_refresh_jti(jti: str, exp: float | None, ttl_fallback_seconds: int) -> bool:
    """Atomically claim a refresh-token JTI.

    Returns False if the JTI was already consumed (replay detected). The
    PRIMARY KEY insert acts as the atomic check-and-set, mirroring Redis
    ``SET NX`` semantics without requiring Redis.
    """
    now = int(datetime.now(timezone.utc).timestamp())
    expires_at = int(exp) if exp else now + ttl_fallback_seconds

    with SessionLocal() as db:
        # Opportunistic purge: rows past their token's expiry are dead weight.
        db.execute(delete(ConsumedRefreshToken).where(ConsumedRefreshToken.expires_at <= now))
        db.add(ConsumedRefreshToken(jti=jti, expires_at=expires_at))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return False
    return True


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