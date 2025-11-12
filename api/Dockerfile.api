# Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal OS packages (kept light)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy pinned runtime requirements for API
COPY requirements-api.txt /app/requirements-api.txt

# Install Python deps (no cache)
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /app/requirements-api.txt

# Copy project files
COPY . /app

# Expose the port used by Uvicorn in Render config
EXPOSE 10000

# Run the FastAPI app
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.main:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120"]
