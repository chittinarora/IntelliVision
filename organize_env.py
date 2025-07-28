#!/usr/bin/env python3
"""
Environment File Organizer - Simple Version
Adds basic grouping comments to your .env file.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def organize_env_file():
    """Organize the .env file with simple grouping comments."""
    env_file = Path('.env')

    if not env_file.exists():
        print("❌ .env file not found!")
        return False

    # Read current .env file
    with open(env_file, 'r') as f:
        current_lines = f.readlines()

    # Parse current variables
    current_vars = {}
    for line in current_lines:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            current_vars[key] = value

    # Create simple organized template
    organized_content = f"""# IntelliVision Environment Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Django Configuration
DJANGO_SECRET_KEY={current_vars.get('DJANGO_SECRET_KEY', 'your-long-random-secret-key-here')}
DJANGO_DEBUG=False
ENVIRONMENT=production

# Database Configuration
POSTGRES_DB={current_vars.get('POSTGRES_DB', 'intellivision')}
POSTGRES_USER={current_vars.get('POSTGRES_USER', 'adminvision')}
POSTGRES_PASSWORD={current_vars.get('POSTGRES_PASSWORD', 'IntelliVisionAIonOS')}
POSTGRES_HOST={current_vars.get('POSTGRES_HOST', 'db')}
POSTGRES_PORT={current_vars.get('POSTGRES_PORT', '5432')}
REDIS_URL={current_vars.get('REDIS_URL', 'redis://redis:6379/0')}

# External Services
MONGO_URI={current_vars.get('MONGO_URI', 'mongodb+srv://toram444444:06nJTevaUItCDpd9@cluster01.lemxesc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster01')}
CLOUDINARY_CLOUD_NAME={current_vars.get('CLOUDINARY_CLOUD_NAME', 'your-cloudinary-cloud-name')}
CLOUDINARY_API_KEY={current_vars.get('CLOUDINARY_API_KEY', 'your-cloudinary-api-key')}
CLOUDINARY_API_SECRET={current_vars.get('CLOUDINARY_API_SECRET', 'your-cloudinary-api-secret')}
AZURE_OPENAI_API_KEY={current_vars.get('AZURE_OPENAI_API_KEY', 'your-azure-openai-api-key')}
ROBOFLOW_API_KEY={current_vars.get('ROBOFLOW_API_KEY', 'your-roboflow-api-key')}

# GPU Configuration (Tesla P100)
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=6.0
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Docker Configuration
SERVICE_TYPE={current_vars.get('SERVICE_TYPE', 'web')}
YOLO_CACHE_DIR={current_vars.get('YOLO_CACHE_DIR', '/app/intellivision/apps/video_analytics/models/yolo_cache')}
JOB_OUTPUT_DIR={current_vars.get('JOB_OUTPUT_DIR', '/app/intellivision/media/outputs')}

# Frontend Configuration
VITE_API_BASE_URL=https://intellivision.aionos.co/api
VITE_MEDIA_BASE_URL=https://intellivision.aionos.co/api/media

# Performance Settings
GPU_MEMORY_FRACTION=0.8
VIDEO_PROCESSING_BATCH_SIZE=64
FRAME_SKIP_THRESHOLD=1
MAX_VIDEO_RESOLUTION=1920x1080
MAX_CONCURRENT_JOBS=10
RATE_LIMIT_REQUESTS=20
RATE_LIMIT_WINDOW=60
CELERY_WORKER_CONCURRENCY=6
CELERY_WORKER_MAX_MEMORY_PER_CHILD=3500000

# Security Settings
FILE_UPLOAD_MAX_MEMORY_SIZE=104857600
DATA_UPLOAD_MAX_MEMORY_SIZE=104857600
"""

    # Create backup
    backup_file = Path('.env.backup')
    with open(backup_file, 'w') as f:
        f.writelines(current_lines)
    print(f"✅ Backup created: {backup_file}")

    # Write organized .env file
    with open(env_file, 'w') as f:
        f.write(organized_content)

    print("✅ .env file organized with simple grouping comments!")
    print("📝 Added groups:")
    print("   - Django Configuration")
    print("   - Database Configuration")
    print("   - External Services")
    print("   - GPU Configuration")
    print("   - Docker Configuration")
    print("   - Frontend Configuration")
    print("   - Performance Settings")
    print("   - Security Settings")

    return True

if __name__ == "__main__":
    success = organize_env_file()
    sys.exit(0 if success else 1)
