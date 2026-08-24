# Multi-stage build: install deps in a builder layer, copy only what's
# needed into the final image, so the shipped image doesn't carry pip's
# build cache / wheel cache (~saves a few hundred MB).

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

FROM python:3.11-slim

WORKDIR /app

# copy installed site-packages from the builder stage instead of
# re-installing (this is the actual multi-stage win)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/
COPY data/policies/ data/policies/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# healthcheck hits our own /health route (checks Redis connectivity +
# whether the RAG pipeline singleton finished loading)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "hr_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
