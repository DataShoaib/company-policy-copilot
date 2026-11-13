from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from hr_rag.api.core.settings import settings


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def _connect_args() -> dict:
    url = settings.database_url or ""
    return {"check_same_thread": False} if url.startswith("sqlite") else {}


def get_engine():
    global _engine
    if _engine is None:
        if not settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Provide a remote PostgreSQL connection "
                "string (e.g. postgresql+psycopg://user:password@host:5432/dbname) "
                "via .env or environment variable."
            )
        _engine = create_engine(
            settings.database_url,
            connect_args=_connect_args(),
            pool_pre_ping=True,
        )
    return _engine


def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


def get_db() -> Generator[Session, None, None]:
    db = get_session_local()()
    try:
        yield db
    finally:
        db.close()