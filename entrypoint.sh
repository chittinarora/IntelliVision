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

# Extract Redis host and port from REDIS_URL
REDIS_HOST=$(echo $REDIS_URL | sed 's/redis:\/\///' | sed 's/:[0-9]*\/.*//')
REDIS_PORT=$(echo $REDIS_URL | sed 's/.*://' | sed 's/\/.*//')

until redis-cli -h $REDIS_HOST -p $REDIS_PORT ping; do
    echo "⏳ Waiting for Redis to be ready..."
    sleep 2
done

echo "✅ Redis connection established"

# =====================================
# Qdrant Connection Check
# =====================================
echo "🔍 Checking Qdrant connection..."

# Extract Qdrant host and port from QDRANT_URL
QDRANT_HOST=$(echo $QDRANT_URL | sed 's/http:\/\///' | sed 's/:[0-9]*//')
QDRANT_PORT=$(echo $QDRANT_URL | sed 's/.*://' | sed 's/\/.*//')

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
python3 intellivision/manage.py migrate

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
            --workers 4 \
            --timeout 300 \
            --keep-alive 2 \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --log-level info | /app/intellivision/log_filter.sh
        ;;
    "celery")
        echo "⚡ Starting Celery worker..."
        exec celery -A intellivision worker \
            --loglevel=warning \
            --concurrency=3 \
            --max-tasks-per-child=1000 \
            --max-memory-per-child=5000000 | /app/intellivision/log_filter.sh
        ;;
    *)
        echo "❌ Unknown service type: $SERVICE_TYPE"
        exit 1
        ;;
esac
