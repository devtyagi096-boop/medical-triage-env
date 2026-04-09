FROM python:3.11-slim

WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY models.py .
COPY environment.py .
COPY grader.py .
COPY baseline.py .
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY uv.lock .
COPY server/ ./server/

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Start server on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]