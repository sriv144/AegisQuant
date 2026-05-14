FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "websockets>=13.0" "alpaca-trade-api>=3.0.0"

# Copy source
COPY . .

# Create data directories
RUN mkdir -p /app/logs /app/data /app/models

# Non-root user for security
RUN useradd -m aegis && chown -R aegis:aegis /app
USER aegis

EXPOSE 8000

# Default: dashboard. Override CMD for the trader daemon.
CMD ["python", "-m", "src.webapp.server", "--host", "0.0.0.0", "--port", "8000"]
