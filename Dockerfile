# Dockerfile for Streamlit frontend
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system packages: keep minimal
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements-streamlit.txt /app/requirements-streamlit.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /app/requirements-streamlit.txt

# copy project
COPY . /app

# Streamlit uses PORT env var on many hosts; default to 8501 locally
ENV PORT=8501

EXPOSE 8501

# start Streamlit (bind to all interfaces)
CMD ["sh", "-c", "streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0"]
