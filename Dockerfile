# ViralOps Engine — Production Dockerfile
# EMADS-PR v1.0 | 16-platform social media automation
#
# Build:  docker build -t viralops-engine .
# Run:    docker run --env-file .env -p 8000:8000 viralops-engine

FROM python:3.13-slim AS base

# ── System deps (ffmpeg for media processing, curl for healthcheck) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps (layer caching — deps change less often) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ──
COPY . .

# ── Create data directories ──
RUN mkdir -p /app/data /app/output /app/logs /app/web/static

# ── Non-root user ──
RUN useradd -m -r viralops && chown -R viralops:viralops /app
USER viralops

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:8000/api/platforms/setup-status || exit 1

# ── Expose port ──
EXPOSE 8000

# ── Start (2 workers for production) ──
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
