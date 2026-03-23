# Use Python 3.11 as requested in README prerequisites
FROM python:3.11-slim

# Install curl and Node.js prerequisites
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Cache-bust: increment CACHE_BUST to force a full rebuild on Railway
# Current value: 4 (bumped 2026-03-23 — SelfEvolvingSystem alias + e2e wiring fixes)
ARG CACHE_BUST=4

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
RUN npm install

# Start command for Railway
# This will run the gateway which spawns the Python backend
CMD ["node", "openclaw-gateway.js"]
