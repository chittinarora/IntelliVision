# Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

################################################################
# Section 1: Install System Dependencies
# - Install build tools and libs (unchanged from original).
# - Added pip cache dir mount for faster rebuilds (uses Docker BuildKit cache).
# - This section is cached if no deps change.
################################################################
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y build-essential libpq-dev gcc cmake git qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools ffmpeg tesseract-ocr postgresql-client && \
    apt-get clean

# Enable BuildKit for better caching (use with DOCKER_BUILDKIT=1 docker build)
# Copy requirements.txt FIRST to cache pip install layer
COPY requirements.txt /app/

################################################################
# Section 2: Install Python Dependencies
# - Upgrade pip and install from requirements.txt.
################################################################
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --find-links https://download.pytorch.org/whl/cu121 --no-cache-dir -r requirements.txt

# Final stage (unchanged, but benefits from cached builder)
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

# Install postgresql-client, curl, and ffmpeg in final stage (ensures all tools are available at runtime)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y postgresql-client curl redis-tools ffmpeg tesseract-ocr && \
    apt-get clean

# Copy backend files
COPY intellivision /app/intellivision
COPY requirements.txt /app/

# Ensure logs, media, and static directories exist
RUN mkdir -p intellivision/logs intellivision/media/outputs intellivision/media/alerts intellivision/media/anpr_outputs intellivision/media/results intellivision/media/uploads intellivision/staticfiles

# Run healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Copy entrypoint script and log filter
COPY entrypoint.sh /app/
COPY intellivision/log_filter.sh /app/intellivision/
RUN chmod +x /app/entrypoint.sh /app/intellivision/log_filter.sh

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
