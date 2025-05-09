FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including Chrome dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ca-certificates \
    build-essential \
    procps \
    git \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Amsterdam
ENV PYTHONPATH=/app:$PYTHONPATH

# Create a directory for environment files
RUN mkdir -p /app/config

# Make scripts executable (with conditional checks to prevent errors)
RUN chmod +x ./docker_setup.sh || echo "docker_setup.sh not found, skipping"
RUN chmod +x ./start.sh || echo "start.sh not found, skipping"
RUN chmod +x ./entrypoint.sh || echo "entrypoint.sh not found, skipping"

# Setup initial configuration
RUN ./docker_setup.sh || echo "docker_setup.sh failed, continuing anyway"

# Expose port for FastAPI
EXPOSE 8000

# Add a script to load environment variables and start the application
RUN echo '#!/bin/sh\n\
if [ -f /app/config/.env ]; then\n\
    export $(cat /app/config/.env | xargs)\n\
fi\n\
exec "$@"\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/start.sh"]
