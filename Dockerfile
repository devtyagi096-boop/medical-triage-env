FROM python:3.11-slim

LABEL maintainer="devtyagi096"
LABEL description="Medical Triage Environment for OpenEnv RL Challenge"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.env.medical_triage_env import MedicalTriageEnv; print('ok')"

EXPOSE 7860

# Default: run the FastAPI server (HuggingFace Spaces)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
