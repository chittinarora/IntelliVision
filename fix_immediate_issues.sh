#!/bin/bash
# IMMEDIATE CRITICAL FIXES SCRIPT
# Run this script to fix critical system stability issues

set -e
echo "🔴 STARTING IMMEDIATE CRITICAL FIXES..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}1. CHECKING MIGRATIONS${NC}"
# Check and apply pending migrations
docker compose exec web python manage.py showmigrations video_analytics || true
echo -e "${YELLOW}Applying pending migrations...${NC}"
docker compose exec web python manage.py migrate video_analytics --verbosity=2

echo -e "${RED}2. FIXING REDIS MEMORY OVERCOMMIT${NC}"
# Enable Redis memory overcommit (requires sudo)
echo -e "${YELLOW}Current overcommit setting:${NC}"
cat /proc/sys/vm/overcommit_memory || echo "Cannot read overcommit setting"

echo -e "${YELLOW}Setting overcommit to 1 (always allow)...${NC}"
echo "NOTE: This requires sudo access. Run manually if script fails:"
echo "sudo sysctl vm.overcommit_memory=1"
echo "sudo sysctl vm.overcommit_ratio=50"

# Try to set overcommit (will fail without sudo)
sudo sysctl vm.overcommit_memory=1 2>/dev/null || echo "⚠️  Run: sudo sysctl vm.overcommit_memory=1"
sudo sysctl vm.overcommit_ratio=50 2>/dev/null || echo "⚠️  Run: sudo sysctl vm.overcommit_ratio=50"

echo -e "${RED}3. FIXING WEB SERVICE OOM SEGFAULT${NC}"
# Create environment overrides for PyTorch memory management
cat > .env.immediate << 'EOF'
# PyTorch Memory Management (prevents OOM during imports)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.6
YOLO_CONFIG_DIR=/tmp/yolo_cache
ULTRALYTICS_SETTINGS_FILE=/tmp/ultralytics.yaml

# Django Migration Safety
DJANGO_MIGRATE_SAFE=1
DJANGO_LAZY_MODELS=1

# Memory Limits
MALLOC_TRIM_THRESHOLD_=100000
MALLOC_MMAP_THRESHOLD_=131072
EOF

echo -e "${YELLOW}Created .env.immediate with memory management settings${NC}"
echo -e "${YELLOW}Add these to your main .env file:${NC}"
cat .env.immediate

echo -e "${RED}4. UPDATING DOCKER COMPOSE FOR IMMEDIATE FIXES${NC}"
# Backup current docker-compose.yml
cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)

# Create immediate fix version
cat > docker-compose.immediate.yml << 'EOF'
# IMMEDIATE FIXES OVERLAY
# Use: docker compose -f docker-compose.yml -f docker-compose.immediate.yml up

version: '3.8'

services:
  web:
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.6
      - YOLO_CONFIG_DIR=/tmp/yolo_cache
      - ULTRALYTICS_SETTINGS_FILE=/tmp/ultralytics.yaml
      - DJANGO_MIGRATE_SAFE=1
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru --save "" --appendonly no
    sysctls:
      - vm.overcommit_memory=1
    deploy:
      resources:
        limits:
          memory: 2.5G
        reservations:
          memory: 1G

  celery:
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
      - CELERYD_MAX_MEMORY_PER_CHILD=3000000
      - CELERY_WORKER_PREFETCH_MULTIPLIER=1
    deploy:
      resources:
        limits:
          memory: 6G
        reservations:
          memory: 3G
EOF

echo -e "${GREEN}5. RESTART SEQUENCE FOR IMMEDIATE FIXES${NC}"
echo -e "${YELLOW}To apply fixes, run:${NC}"
echo "1. docker compose down"
echo "2. docker compose -f docker-compose.yml -f docker-compose.immediate.yml up -d"
echo "3. docker compose logs -f web celery redis"

echo -e "${GREEN}6. VERIFICATION COMMANDS${NC}"
echo -e "${YELLOW}After restart, verify fixes:${NC}"
echo "# Check migrations applied:"
echo "docker compose exec web python manage.py showmigrations video_analytics"
echo ""
echo "# Check Redis memory settings:"
echo "docker compose exec redis redis-cli config get maxmemory"
echo ""
echo "# Check web service memory:"
echo "docker compose exec web ps aux"
echo ""
echo "# Monitor for segfaults:"
echo "docker compose logs web | grep -i 'segmentation\|signal\|core'"

echo -e "${GREEN}✅ IMMEDIATE FIXES SCRIPT COMPLETED${NC}"
echo -e "${YELLOW}Next: Run GPU optimization fixes after these are stable${NC}"
EOF