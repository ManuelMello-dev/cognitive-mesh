FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with error handling
RUN pip install --no-cache-dir -r requirements.txt || \
    (pip install --no-cache-dir aiohttp pyzmq yfinance requests python-dotenv pydantic structlog msgpack mmh3 && \
     echo "Core dependencies installed, optional DB packages may have failed")

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('localhost', 5555), timeout=5)" || exit 1

# Run the application
CMD ["python", "main.py"]
