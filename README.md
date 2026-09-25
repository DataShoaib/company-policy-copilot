# Company Policy Copilot

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers employee questions about company HR policies — with role-based access control, guardrails, caching, and full observability.

Built for **TechCorp India Pvt. Ltd.** (fictional organization used for policy content).

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-DC244C)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

---

## Overview

HR Policy Copilot lets employees log in, ask natural-language questions about HR policy, and get answers grounded strictly in the company's policy documents — scoped to what their role is allowed to see.

**How a query flows through the system:**

1. User logs in and gets a JWT (role embedded in the token).
2. Question passes through an input guardrail (regex + LLM judge) to catch injections, off-topic queries, and PII.
3. A Redis cache is checked using a **role-scoped** cache key (no cross-role leakage).
4. The question is routed to one or more of 9 policy categories, restricted to what the user's role can access.
5. Relevant chunks are retrieved from Qdrant.
6. An LLM generates a grounded answer from the retrieved context.
7. The answer passes through an output guardrail (PII-leak + hallucination check) before being cached and returned.
8. The Streamlit UI displays the answer, its sources, latency, and cache status.

---

## Features

### Authentication & User Management
- Self-service signup (always provisioned as `employee` — role cannot be self-selected)
- JWT-based login: 30-minute access token + 7-day refresh token
- Token refresh endpoint
- Admin-only user provisioning (`hr_admin` can create users with any role)
- **Retrieval-level RBAC** — access is enforced by restricting which Qdrant collections are searched, not just by blocking routes. An `employee` can never retrieve finance/legal documents, even indirectly.

### Query Pipeline
A 7-stage pipeline for every question:
1. Regex input guardrail (empty input, prompt-injection patterns, PII patterns — Aadhaar, PAN, phone, email)
2. Redis cache lookup (role-scoped key, ~5ms on hit)
3. LLM-based input guardrail (classifies as safe / prompt injection / off-topic / PII) — fails open if the LLM is unreachable
4. Category routing (explicit category if provided and allowed, otherwise keyword-based routing across all allowed categories)
5. Retrieval from Qdrant (single-category direct fetch, or round-robin merge across categories, top-3 results)
6. Answer generation with a strict, anti-refusal prompt and automatic retry (2 attempts, exponential backoff)
7. Output guardrails: PII-leak check and an LLM hallucination judge

### Caching
- Redis-backed answer cache, keyed by a hash of `scope | category | question`
- Role scope is part of the cache key, so a cached answer can never leak across roles
- 1-hour TTL (configurable)
- Only non-empty, sourced answers are cached — refusals are always evaluated fresh

### Rate Limiting
- 20 requests/minute per user on the query endpoint
- 5 requests/minute on login (brute-force protection)
- Redis-backed, safe across multiple workers
- Toggleable via configuration

### Frontend (Streamlit)
- Login screen with configurable backend URL
- Role badge, category selector, and chat interface
- One-click example questions for smoke testing
- Expandable source citations (category, document ID, title, snippet)
- Latency and cache-hit indicators
- Admin panel for provisioning new users (visible only to `hr_admin`)

### Refusal Handling
Refusals are returned as **HTTP 200**, not 403 — with a polite message and (where relevant) supporting sources, so the frontend doesn't need special error handling. Three refusal types are supported: out-of-role-scope, no matching documents, and LLM-judged "not covered by policy."

### Evaluation & Testing
- 13-question golden evaluation set (currently 13/13 pass rate on the production prompt)
- 61 automated tests covering routes, RBAC, caching, guardrails, the pipeline, and the frontend
- Notebooks for retrieval experiments (e.g., BM25 vs. vector search)
- Linted with Ruff on every commit

### Observability
- **LangSmith** tracing — one root span per query, with child spans for retrieval, generation, and guardrail checks
- **MLflow** for experiment tracking, with optional **DagsHub** remote backend
- **Ragas** for RAG evaluation metrics

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language / Runtime | Python 3.10–3.13, Uvicorn (ASGI) |
| Backend API | FastAPI, Pydantic v2, `pydantic-settings` |
| Frontend | Streamlit |
| LLM Gateway | LiteLLM Router — Groq (primary) with Gemini fallback, via LangChain `ChatLiteLLM` |
| Vector Store | Qdrant (Docker or local file fallback), 9 category-scoped collections |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local) |
| Chunking | `langchain-text-splitters`, 500-token chunks / 50-token overlap |
| Auth | JWT (`python-jose`, HS256), `passlib[bcrypt]` |
| Database | SQLite (dev) or PostgreSQL (Docker) via SQLAlchemy 2.0 |
| Cache / Rate Limiting | Redis |
| Tracing / Evaluation | LangSmith, MLflow, DagsHub, Ragas |
| Infra | Docker Compose (Qdrant, Redis, Postgres, API), Ruff, Pytest |

---

## Architecture

```
                         ┌─────────────────┐
                         │   Streamlit UI   │  (port 8502)
                         └────────┬─────────┘
                                  │ REST
                         ┌────────▼─────────┐
                         │   FastAPI API     │  (port 8001)
                         │  auth · query ·   │
                         │  health           │
                         └────────┬─────────┘
                                  │
        ┌─────────────┬──────────┼──────────┬──────────────┐
        ▼             ▼          ▼          ▼              ▼
   Input Guard   Redis Cache  Router   Qdrant Retrieval  Output Guard
   (regex+LLM)   (role-scoped) (keyword) (9 collections)  (PII+halluc.)
                                  │
                                  ▼
                         LiteLLM Router
                         Groq → Gemini fallback
```

Policy categories (9 total, each its own Qdrant collection):
`leave` · `compensation` · `conduct` · `performance` · `recruitment` · `finance` · `it` · `legal` · `operations`

---

## Project Structure

```
hr-policy-copilot/
├── .env.example                 # Environment variable template
├── docker-compose.yml           # qdrant + redis + postgres + api
├── pyproject.toml               # Dependencies, Ruff config, Pytest config
├── run-hr.bat                   # Windows one-click launcher
│
├── data/
│   ├── policies/                # 9 markdown policy documents (HRP-001 … HRP-009)
│   ├── eval/                    # 13-question evaluation set
│   ├── hr_policy.db             # SQLite users DB (dev)
│   └── qdrant/                  # Local Qdrant fallback storage
│
├── docker/
│   └── Dockerfile               # API image
│
├── frontend/
│   └── app.py                   # Streamlit UI (login, chat, admin panel)
│
├── src/hr_rag/
│   ├── config.py                # Constants, models, categories, tracing setup
│   ├── pipeline.py              # Retrieval + generation chains, Qdrant verification
│   ├── prompts.py                # Answer prompt (anti-refusal), HyDE / rewrite prompts
│   ├── llm.py                    # LiteLLM Router (Groq → Gemini)
│   ├── embeddings.py            # Embedding model loader (singleton)
│   ├── chunking.py               # Document chunking
│   ├── data_loading.py          # Loads policy docs + metadata
│   ├── formatting.py             # Formats retrieved docs into context
│   ├── qdrant_store.py          # Qdrant collection management
│   ├── retrievers/
│   │   └── router.py             # Keyword-based category router
│   └── api/
│       ├── main.py               # App factory, lifespan, CORS, routers
│       ├── core/
│       │   ├── settings.py       # Environment-driven settings
│       │   ├── security.py       # JWT + password hashing
│       │   ├── rbac.py           # Role → allowed categories
│       │   ├── redis_client.py   # Redis singleton
│       │   └── rate_limit.py     # Redis-backed rate limiter
│       ├── routes/
│       │   ├── auth.py           # signup / login / refresh / provision
│       │   ├── query.py          # POST /query
│       │   └── health.py         # GET /health
│       └── services/
│           ├── rag_service.py    # 7-stage pipeline orchestrator
│           ├── guardrails.py     # Input/output guardrails
│           └── cache.py          # Cache get/set, scope fingerprinting
│
├── tests/                        # 61 automated tests
└── notebooks/                    # Retrieval experiments
```

---

## Results & Benchmarks

### Performance
| Metric | Value |
|---|---|
| Median query latency (cache miss) | ~2s (p50), ~5s (p95 — varies with LLM provider) |
| Cache-hit latency | ~5ms |
| Test suite | 61/61 passing |
| Anti-refusal prompt eval | 13/13 |
| Linting | Ruff clean |

### Retrieval Strategy Experiments
Six retrieval strategies were benchmarked on a 5-question, category-balanced eval set and scored on 5 RAGAS metrics, with every run tracked in MLflow and traced end-to-end in LangSmith:

| Strategy | Faithfulness | Correctness | Latency | Outcome |
|---|---|---|---|---|
| **Baseline (dense retrieval)** | **0.83** | **0.76** | ~37s | **Selected for production** |
| Hybrid (BM25 + dense) | 0.50 | — | — | Rejected — faithfulness regression |
| Contextual compression | — | — | ~392s | Rejected — ~10× slower than baseline |
| Query rewrite | — | — | — | Evaluated, not selected |
| HyDE | — | — | — | Evaluated, not selected |
| Metadata filtering | — | — | — | Evaluated, not selected |

Baseline dense retrieval outperformed every added-complexity strategy on faithfulness/correctness while staying fastest, and was shipped to production as-is.

---

## Getting Started

### Prerequisites
- Python 3.10–3.13
- Docker & Docker Compose
- API keys for Groq and Google (Gemini)

### 1. Clone and configure
```bash
git clone https://github.com/DataShoaib/company-policy-copilot.git
cd company-policy-copilot
cp .env.example .env
# Fill in GROQ_API_KEY, GOOGLE_API_KEY, LANGSMITH_API_KEY, etc.
```

### 2. Start infrastructure
```bash
docker-compose up -d qdrant redis postgres
```

### 3. Install dependencies
```bash
pip install -e .
```

### 4. Run the backend
```bash
uvicorn src.hr_rag.api.main:app --reload --port 8001
```

### 5. Run the frontend
```bash
streamlit run frontend/app.py --server.port 8502 -- --api-url http://localhost:8001
```

### 6. Try it out
- API docs: `http://localhost:8001/docs`
- Health check: `http://localhost:8001/health`
- UI: `http://localhost:8502`

> On Windows, `run-hr.bat` starts everything with one click.

---

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` / `GROQ_MODEL` | Primary LLM provider |
| `GOOGLE_API_KEY` / `GOOGLE_MODEL` | Fallback LLM provider |
| `DATABASE_URL` | SQLite (dev) or PostgreSQL (Docker) connection string |
| `REDIS_URL` | Cache and rate-limit backend |
| `QDRANT_URL` | Vector DB endpoint (leave empty to use local file fallback) |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | Tracing |
| `JWT_SECRET` | **Must be overridden in production** |
| `CACHE_TTL_SECONDS` | Answer cache TTL (default 3600) |
| `RATE_LIMIT_PER_MINUTE` | Query rate limit |
| `LOGIN_RATE_LIMIT_PER_MINUTE` | Login rate limit |

---

## Testing

```bash
pytest
ruff check .
```

61 tests cover authentication, RBAC enforcement, caching, guardrails, the retrieval pipeline, and the Streamlit frontend (via `AppTest`).

---

## Known Limitations

- No user deprovisioning endpoint — accounts cannot be disabled or deleted via the API.
- LLM request timeout is currently set high (600s); tightening this is a planned optimization.
- `RERANK_CANDIDATE_K` is larger than strictly necessary for the current 9-document corpus.
- CORS is currently limited to `localhost:8501`; the frontend's actual port (`8502`) needs to be added to avoid browser CORS warnings.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Author

**Shoaib** ([@DataShoaib](https://github.com/DataShoaib))
