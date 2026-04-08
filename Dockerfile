# ── ContentModerationEnv — Hugging Face Spaces Dockerfile ────────────────────
# SDK: Docker  (set sdk: docker in README.md YAML block)
# Port: 7860 (HF Spaces requirement)

FROM python:3.11-slim

# ── system deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# ── non-root user (HF Spaces security requirement) ────────────────────────────
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── application files ─────────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .
COPY --chown=appuser:appuser inference.py .

USER appuser

# ── environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# ── startup ───────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
