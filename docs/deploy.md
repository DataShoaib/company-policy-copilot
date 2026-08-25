# Render blueprint for the Policy Copilot API.
# Pair it with the managed services below (or BYO Redis/Qdrant/Postgres).
#
#   render.yaml -> Render infra-as-code. Save as render.yaml in the repo root
#   and connect the repo; Render provisions the services + environment.
#
# services:
# # Managed backing services (one-click in Render dashboard) OR external.
#   - type: redis        # Render Redis
#   - type: qdrant       # Render Qdrant Cloud paid tier, or use an external URL
#   - type: postgres     # Render Postgres, then set DATABASE_URL
#
#   - type: web
#     name: hr-policy-api
#     runtime: python
#     plan: free
#     buildCommand: pip install -e .
#     startCommand: uvicorn hr_rag.api.main:app --host 0.0.0.0 --port $PORT
#     healthCheckPath: /health
#     envVars:
#       - key: JWT_SECRET_KEY
#         sync: false   # generate via `openssl rand -hex 32`
#       - key: GROQ_API_KEY
#         sync: false
#       - key: GOOGLE_API_KEY
#         sync: false
#       - key: LANGCHAIN_API_KEY
#         sync: false
#       - key: REDIS_URL
#         value: redis://...      # managed Redis URL
#       - key: QDRANT_URL
#         value: https://...       # managed Qdrant cluster URL
#       - key: DATABASE_URL
#         value: postgresql+psycopg://...  # managed Postgres

# Railway quickstart (alternative/one-file deploy):
#   - README tells you to add a Railway service from this repo with:
#     start command: uvicorn hr_rag.api.main:app --host 0.0.0.0 --port $PORT
#     + set the env vars (JWT_SECRET_KEY, GROQ_API_KEY, GOOGLE_API_KEY,
#       LANGCHAIN_API_KEY, REDIS_URL, QDRANT_URL, DATABASE_URL)
#     + attach Railway Postgres/Redis plugins (or a managed Qdrant cluster).