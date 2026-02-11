# ViralOps Engine — Production Dockerfile
# Multi-stage build for minimal image size

FROM python:3.13-slim AS base

# ── System deps ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ──
COPY . .

# ── Create data directory ──
RUN mkdir -p /app/data /app/web/static

# ── Non-root user ──
RUN useradd -m -r viralops && chown -R viralops:viralops /app
USER viralops

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/health'); assert r.status_code == 200"

# ── Expose port ──
EXPOSE 8000

# ── Start ──
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
