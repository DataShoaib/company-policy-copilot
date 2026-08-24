# HR Policy Copilot

![CI](https://github.com/datashoaib/hr-policy-copilot/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

RAG system that answers employee questions from company policy docs across HR, Finance, IT, Legal, and Operations. Built this because employees ask the same policy questions repeatedly, so a grounded self-serve bot saves real time.

## What's here

- `notebooks/experiments.ipynb` — compares baseline, hybrid, query rewrite, multi-query, compression, metadata filtering, HyDE and cross-encoder retrieval on a small category-balanced evaluation sample, scored with RAGAS. The notebook uses the same Qdrant collections as the API.
- `src/hr_rag/` — the actual retrieval/RAG library, shared between the notebook and the API.
- `src/hr_rag/api/` — FastAPI service with category-scoped Qdrant retrieval, JWT auth, PostgreSQL-backed users, role-based access, Redis caching and rate limiting.
- `frontend/app.py` — Streamlit client with login and a department selector that sends the selected category to one Qdrant collection.
- **Observability** — every request's prompt/LLM run is traced to LangSmith (project `HR-RAG-Experiments`: inputs, outputs, token usage, latency); MLflow tracks the retrieval experiments.

## Why it's built this way

The whole point of this tool is that the same handful of questions get asked repeatedly, on a corpus that barely changes. That's what drove most of the decisions:

- **Category-scoped Qdrant retrieval** — each department/subcategory has its own collection, so a finance question does not search IT or Legal. Qdrant payload metadata supports additional filters such as document version or tenant.
- **Deterministic query routing** — explicit categories are validated first, then known keywords select relevant allowed collections. Ambiguous questions search only the user's allowed collections.
- **JWT + role-based access** — not all policies are equally sensitive. Compensation has salary bands and bonus formulas that a regular employee shouldn't be pulling up for other grades. Enforced at the retrieval layer, not just the route, so it can't be bypassed by a forgotten check somewhere else.
- **Redis caching** — the normalized question is the cache key, so identical questions reuse the same LLM output. Before reuse, source categories are checked against the current user's permissions.
- **Rate limiting** — per user, not per IP (this sits behind auth already, and IP-based limiting breaks behind office VPN/NAT anyway).
- **Retry, not a circuit breaker** — a couple of retries on a flaky LLM call is enough here. A real circuit breaker matters at a traffic scale this internal tool doesn't have.

Didn't add an agentic multi-hop loop either — most of these questions are single-hop lookups. There's a `cross-policy` slice in the eval set specifically to check multi-hop questions against plain retrieval, and it holds up fine with a decent prompt.

## Layout

```
data/policies/       9 source policy docs across HR and four departments
data/eval/           curated multi-department eval set (category, difficulty, question type)
notebooks/           experiments.ipynb -- retrieval comparisons and RAGAS evaluation
src/hr_rag/          retrieval library (chunking, embeddings, Qdrant, routing, pipeline)
src/hr_rag/api/       FastAPI app -- auth, rbac, caching, rate limiting, routes
scripts/ingest.py    builds the Qdrant collections offline
tests/               pytest, no API key needed
docker/, docker-compose.yml
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env      # add your GROQ_API_KEY (console.groq.com/keys) and a JWT_SECRET_KEY
python scripts/ingest.py  # builds Qdrant collections once
```

Start the backend and its infrastructure separately:

```bash
  docker compose up --build          # starts PostgreSQL + Qdrant + Redis + FastAPI
```

In a second terminal, start the separate Streamlit frontend directly:

```bash
streamlit run frontend/app.py
```

The frontend opens at `http://localhost:8501` and calls the backend at `http://localhost:8000`. Set `API_URL` when the backend is hosted elsewhere.

or locally, with Redis running separately:

```bash
uvicorn hr_rag.api.main:app --reload --port 8000
```

Docs at `http://localhost:8000/docs`.

For the notebook (uses 10 questions by default to stay within free-tier limits):

```bash
pip install -e ".[dev]"
jupyter notebook notebooks/experiments.ipynb
```

Tests:

```bash
pytest tests/ -v
```

## Trying the API

Three demo users are seeded into PostgreSQL on first startup:

- `employee1` / `employee123` -- sees leave, conduct, recruitment
- `manager1` / `manager123` -- + performance
- `hradmin1` / `hradmin123` -- all categories

```bash
curl -X POST localhost:8000/auth/login -H "Content-Type: application/json" \
  -d '{"username": "employee1", "password": "employee123"}'

curl -X POST localhost:8000/query -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of casual leave do I get?"}'
```

Ask the same question again and `cached` flips to `true`, latency drops to a few ms. Ask a compensation question as `employee1` and it'll say it doesn't have access -- log in as `hradmin1` and it answers.

## What's not in here yet

Circuit breaker + LLM provider fallback, a proper CI eval gate, external identity provider, production secret manager, and full observability dashboards. These are the next steps for a larger deployment; the current project is production-style for portfolio use.
