#!/usr/bin/env python3
"""
Environment Configuration Checker
Safely checks your .env file for required variables without exposing sensitive data.
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check .env file for required variables."""
    env_file = Path('.env')

    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with the required variables.")
        return False

    # Load environment variables
    with open(env_file, 'r') as f:
        env_content = f.read()

    # Define required variables
    required_vars = {
        'DJANGO_SECRET_KEY': 'Django secret key (required for security)',
        'VITE_API_BASE_URL': 'Frontend API base URL (required for Apache)',
        'MONGO_URI': 'MongoDB connection string (required for car counting)',
        'CLOUDINARY_CLOUD_NAME': 'Cloudinary cloud name (required for image storage)',
        'CLOUDINARY_API_KEY': 'Cloudinary API key (required for image storage)',
        'CLOUDINARY_API_SECRET': 'Cloudinary API secret (required for image storage)',
        'AZURE_OPENAI_API_KEY': 'Azure OpenAI API key (required for food waste)',
        'ROBOFLOW_API_KEY': 'Roboflow API key (required for pothole detection)',
    }

    # Define GPU variables
    gpu_vars = {
        'CUDA_VISIBLE_DEVICES': 'GPU device selection',
        'TORCH_CUDA_ARCH_LIST': 'CUDA architecture',
        'NVIDIA_VISIBLE_DEVICES': 'NVIDIA GPU access',
        'NVIDIA_DRIVER_CAPABILITIES': 'NVIDIA driver capabilities',
    }

    # Define database variables
    db_vars = {
        'POSTGRES_DB': 'PostgreSQL database name',
        'POSTGRES_USER': 'PostgreSQL username',
        'POSTGRES_PASSWORD': 'PostgreSQL password',
        'POSTGRES_HOST': 'PostgreSQL host',
        'POSTGRES_PORT': 'PostgreSQL port',
        'REDIS_URL': 'Redis connection URL',
    }

    print("🔍 Checking .env configuration...")
    print("=" * 50)

    # Check required variables
    print("\n📋 Required Variables:")
    missing_required = []
    for var, description in required_vars.items():
        if var in env_content:
            print(f"✅ {var}: {description}")
        else:
            print(f"❌ {var}: {description} - MISSING")
            missing_required.append(var)

    # Check GPU variables
    print("\n🎮 GPU Configuration:")
    missing_gpu = []
    for var, description in gpu_vars.items():
        if var in env_content:
            print(f"✅ {var}: {description}")
        else:
            print(f"⚠️  {var}: {description} - MISSING")
            missing_gpu.append(var)

    # Check database variables
    print("\n🗄️  Database Configuration:")
    missing_db = []
    for var, description in db_vars.items():
        if var in env_content:
            print(f"✅ {var}: {description}")
        else:
            print(f"⚠️  {var}: {description} - MISSING")
            missing_db.append(var)

    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")

    if missing_required:
        print(f"❌ Missing {len(missing_required)} required variables:")
        for var in missing_required:
            print(f"   - {var}")
    else:
        print("✅ All required variables are set!")

    if missing_gpu:
        print(f"⚠️  Missing {len(missing_gpu)} GPU variables (may affect performance)")

    if missing_db:
        print(f"⚠️  Missing {len(missing_db)} database variables")

    # Check for production settings
    print("\n🔧 Production Settings:")
    if 'DJANGO_DEBUG=False' in env_content:
        print("✅ DJANGO_DEBUG=False (production mode)")
    else:
        print("⚠️  DJANGO_DEBUG not set to False (should be False for production)")

    if 'ENVIRONMENT=production' in env_content:
        print("✅ ENVIRONMENT=production")
    else:
        print("⚠️  ENVIRONMENT not set to production")

    if 'VITE_API_BASE_URL=https://intellivision.aionos.co/api' in env_content:
        print("✅ VITE_API_BASE_URL set for production")
    else:
        print("⚠️  VITE_API_BASE_URL not set for production domain")

    return len(missing_required) == 0

if __name__ == "__main__":
    success = check_env_file()
    sys.exit(0 if success else 1)
