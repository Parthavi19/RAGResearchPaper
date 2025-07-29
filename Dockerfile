# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY . .

# Verify gunicorn.conf.py exists
RUN if [ ! -f /app/gunicorn.conf.py ]; then echo "Error: gunicorn.conf.py not found"; exit 1; fi

# Create required directories
RUN mkdir -p /app/uploads /app/local_qdrant

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user and set ownership
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose application port
EXPOSE 8080

# Health check - Extended start-period for ML model loading
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Start the application with Gunicorn
CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:application"]

# Start the application with Gunicorn
CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:application"]
