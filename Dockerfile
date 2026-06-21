FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create data directory
RUN mkdir -p data logs

# Run the bot
CMD ["python", "-u", "src/nq_sentiment.py"]