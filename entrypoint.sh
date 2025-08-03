#!/bin/bash

# =====================================
# IntelliVision Docker Entrypoint
# =====================================
# This script handles container startup, database migrations, and service initialization.
# Added GPU runtime validation and enhanced error handling.

set -e

# =====================================
# Environment Setup
# =====================================
echo "🚀 Starting IntelliVision container..."

# Set default environment variables
export PYTHONPATH=${PYTHONPATH:-/app/intellivision:/app}
export DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE:-intellivision.settings}

# =====================================
# GPU Runtime Validation
# =====================================
echo "🔍 Checking GPU runtime configuration..."

# Check if NVIDIA runtime is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU not detected - running on CPU"
fi

# Check if CUDA is available in Python
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('⚠️  CUDA not available - using CPU')
"

# =====================================
# Database Connection Check
# =====================================
echo "🗄️  Checking database connection..."

# Wait for database to be ready
until pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB; do
    echo "⏳ Waiting for database to be ready..."
    sleep 2
done

echo "✅ Database connection established"

# =====================================
# Redis Connection Check
# =====================================
echo "🔴 Checking Redis connection..."

# Parse Redis URL using Python (more robust than sed)
REDIS_PARTS=$(python3 -c "
from urllib.parse import urlparse
url = '$REDIS_URL'
result = urlparse(url)
print(f'{result.hostname} {result.port or 6379}')
")
read -r REDIS_HOST REDIS_PORT <<< "$REDIS_PARTS"

# Fallback to defaults if parsing fails
REDIS_HOST=${REDIS_HOST:-redis}
REDIS_PORT=${REDIS_PORT:-6379}

until redis-cli -h $REDIS_HOST -p $REDIS_PORT ping; do
    echo "⏳ Waiting for Redis to be ready..."
    sleep 2
done

echo "✅ Redis connection established"

# =====================================
# Qdrant Connection Check
# =====================================
echo "🔍 Checking Qdrant connection..."

# Parse Qdrant URL using Python (more robust than sed)
QDRANT_PARTS=$(python3 -c "
from urllib.parse import urlparse
url = '$QDRANT_URL'
result = urlparse(url)
print(f'{result.hostname} {result.port or 6333}')
")
read -r QDRANT_HOST QDRANT_PORT <<< "$QDRANT_PARTS"

# Fallback to defaults if parsing fails
QDRANT_HOST=${QDRANT_HOST:-qdrant}
QDRANT_PORT=${QDRANT_PORT:-6333}

until curl -f http://$QDRANT_HOST:$QDRANT_PORT/collections; do
    echo "⏳ Waiting for Qdrant to be ready..."
    sleep 2
done

echo "✅ Qdrant connection established"

# =====================================
# Django Setup
# =====================================
echo "🐍 Setting up Django..."

# Create necessary directories (mounted from host)
mkdir -p /app/intellivision/logs
mkdir -p /app/intellivision/media/outputs
mkdir -p /app/intellivision/media/alerts
mkdir -p /app/intellivision/media/anpr_outputs
mkdir -p /app/intellivision/media/results
mkdir -p /app/intellivision/media/uploads
mkdir -p /app/intellivision/staticfiles

# Collect static files
echo "📦 Collecting static files..."
python3 intellivision/manage.py collectstatic --noinput

# Run database migrations
echo "🔄 Running database migrations..."
python3 intellivision/manage.py migrate --noinput

# Check for pending migrations (optional, for debugging)
echo "🔍 Checking migration status..."
python3 intellivision/manage.py showmigrations --verbosity=0

# =====================================
# Service Type Detection
# =====================================
SERVICE_TYPE=${SERVICE_TYPE:-web}

echo "🎯 Starting service type: $SERVICE_TYPE"
echo "📋 Logs will be filtered to show only important events (video analytics, errors, system events)"

case $SERVICE_TYPE in
    "web")
        echo "🌐 Starting Django web server..."
        exec /usr/local/bin/gunicorn intellivision.wsgi:application \
            --bind 0.0.0.0:8001 \
            --workers 1 \
            --timeout 300 \
            --keep-alive 2 \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --log-level info | /app/intellivision/log_filter.sh
        ;;
    "celery")
        echo "⚡ Starting Celery worker..."
        exec celery -A intellivision worker \
            --loglevel=info \
            -Q gpu_queue,celery \
            --concurrency=3 \
            --max-tasks-per-child=1000 \
            --max-memory-per-child=5000000
        ;;
    *)
        echo "❌ Unknown service type: $SERVICE_TYPE"
        exit 1
        ;;
esac
