import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hr_rag.api.core.database import init_db
from hr_rag.api.core.settings import settings
from hr_rag.api.core.users import seed_demo_users
from hr_rag.api.routes import auth, health, query
from hr_rag.config import GROQ_MODEL, enable_langsmith_tracing
from hr_rag.pipeline import get_pipeline

_JWT_PLACEHOLDER_PREFIXES = ("CHANGE_ME", "your_jwt", "replace_me")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Security: refuse to boot on a placeholder JWT secret. Otherwise an
    # attacker can forge valid access tokens signed with the well-known default.
    if settings.jwt_secret_key.startswith(_JWT_PLACEHOLDER_PREFIXES):
        raise RuntimeError(
            "JWT_SECRET_KEY is still the placeholder default. Generate a real one "
            "with 'openssl rand -hex 32' and set it in .env before running."
        )

    # DB tables + demo users are created here, not at import time, so importing
    # the app for tests/tooling does not trigger writes as a side effect.
    init_db()
    seed_demo_users()

    # Enable LangSmith BEFORE warming the pipeline so anything that runs
    # during startup is traced too, not just request-time calls.
    enable_langsmith_tracing()

    # Warm the embedding model + collections now, but don't let a missing or
    # unconfigured LLM key take the whole API down -- /health can report
    # pipeline_loaded=false ("degraded") instead.
    try:
        get_pipeline()
    except Exception as exc:  # noqa: BLE001 - startup must not fail on provider config
        print(f"[startup] WARNING: RAG pipeline failed to initialize: {exc}", file=sys.stderr)

    print(f"RAG model configured: {GROQ_MODEL}")
    yield


app = FastAPI(
    title="HR Policy Copilot API",
    description="JWT-secured, RBAC-scoped, Redis-cached RAG API for TechCorp HR policy Q&A.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(query.router)


@app.get("/")
def root():
    return {"service": "HR Policy Copilot API", "docs": "/docs", "health": "/health"}