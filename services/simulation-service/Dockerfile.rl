# Base image with all dependencies - this will be cached
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Final image that gets the latest code
FROM base as final

WORKDIR /app

# Copy Python modules and model files
COPY *.py /app/
COPY hourly\(1\).csv /app/
COPY weight.zip /app/

# Expose port
EXPOSE 5000

# Run the script directly
CMD ["python", "/app/rl_service.py"]
