# Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential libpq-dev gcc cmake git qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools ffmpeg tesseract-ocr && \
    apt-get clean

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/lib /usr/lib

# Copy backend files
COPY intellivision /app/intellivision
COPY requirements.txt /app/

# Remove nested directories
RUN rm -rf /app/intellivision/intelli-vision /app/intellivision/intellivision

# Ensure logs, media, and static directories exist
RUN mkdir -p intellivision/logs intellivision/media/outputs intellivision/media/alerts intellivision/media/anpr_outputs intellivision/media/results intellivision/media/uploads intellivision/staticfiles

# Collect static files
RUN python intellivision/manage.py collectstatic --noinput

# Run healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Copy entrypoint script
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]