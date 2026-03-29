FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY models.py .
COPY environment.py .
COPY grader.py .
COPY baseline.py .
COPY openenv.yaml .
COPY server/ ./server/

# Expose port
EXPOSE 8000

# Run the server (no healthcheck - let HF handle it)
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]