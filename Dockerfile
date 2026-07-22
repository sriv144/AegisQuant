FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the reviewed Python 3.11 lockfile (cached layer).
COPY requirements.lock .
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.lock \
    && pip check

# Copy source
COPY . .

# Create data directories
RUN mkdir -p /app/logs /app/data /app/models

# Non-root user for security
RUN useradd -m aegis && chown -R aegis:aegis /app
USER aegis

EXPOSE 8000

# The image has no default trading process.
CMD ["python", "-m", "src.webapp.server"]
