# /intellivision/intellivision/settings.py

"""
=====================================
Django Settings
=====================================
Configuration for the IntelliVision Django project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from corsheaders.defaults import default_headers
from datetime import timedelta
import json
import logging

# Custom JSON formatter class
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "process": record.process,
            "thread": record.thread,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }
        return json.dumps(log_entry)

"""
=====================================
Path & Environment
=====================================
Sets up base directory and loads environment variables.
Removed duplicate BASE_DIR definition (Issue #5).
"""

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')  # Loads .env from project root

"""
=====================================
Security
=====================================
Configures security settings, including SSL and HSTS.
"""

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("DJANGO_SECRET_KEY environment variable is required")
DEBUG = os.environ.get("DJANGO_DEBUG", "False") == "True"

# Only redirect to HTTPS in production
if not DEBUG:
    SECURE_SSL_REDIRECT = True
else:
    SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = True  # Related to Issue #3, kept as is per user instruction
CSRF_COOKIE_SECURE = True    # Related to Issue #3, kept as is per user instruction
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_REFERRER_POLICY = "strict-origin"
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
ALLOWED_HOSTS = ['intellivision.aionos.co', '34.100.200.148', '35.190.199.96', 'localhost', '127.0.0.1']

"""
=====================================
Database
=====================================
Configures PostgreSQL with connection reuse for performance.
"""

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('POSTGRES_DB') or 'intellivision',  # Safeguard for empty; uses fallback if empty/unset
        'USER': os.environ.get('POSTGRES_USER') or 'adminvision',  # Similar safeguard
        'PASSWORD': os.environ.get('POSTGRES_PASSWORD') or 'IntelliVisionAIonOS',
        'HOST': os.environ.get('POSTGRES_HOST') or 'db',
        'PORT': os.environ.get('POSTGRES_PORT') or '5432',
        'CONN_MAX_AGE': 600,
    }
}

"""
=====================================
Application Definition
=====================================
Defines installed apps, including Django, DRF, and CORS headers.
"""

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'apps.video_analytics',
    'apps.face_auth',
]

"""
=====================================
Middleware
=====================================
Configures middleware, ensuring CORS is handled first.
"""

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

"""
=====================================
Templates
=====================================
Configures Django templates.
"""

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

"""
=====================================
WSGI/ASGI
=====================================
Configures WSGI application.
"""

ROOT_URLCONF = 'intellivision.urls'
WSGI_APPLICATION = 'intellivision.wsgi.application'

"""
=====================================
Static & Media Files
=====================================
Configures storage for static and media files, optimized for Docker.
Updated MEDIA_ROOT to use Docker-compatible path.
"""

STATIC_URL = '/api/static/'
STATIC_ROOT = '/app/intellivision/staticfiles'  # Docker-compatible path
MEDIA_URL = '/api/media/'
MEDIA_ROOT = '/app/intellivision/media'  # Docker-compatible path
DEFAULT_FILE_STORAGE = 'django.core.files.storage.FileSystemStorage'

"""
=====================================
Analytics Output Directory
=====================================
Configures directory for analytics job outputs.
Updated to use Docker-compatible path.
"""

JOB_OUTPUT_DIR = os.environ.get('JOB_OUTPUT_DIR', '/app/intellivision/media/outputs')
os.makedirs(JOB_OUTPUT_DIR, exist_ok=True)

"""
=====================================
Analytics Temp Directory
=====================================
Configures directory for temporary files during analytics processing.
Uses persistent storage within media directory for better reliability.
"""

JOB_TEMP_DIR = os.environ.get('JOB_TEMP_DIR', '/app/intellivision/media/temp')
os.makedirs(JOB_TEMP_DIR, exist_ok=True)

"""
=====================================
Authentication & Password Validation
=====================================
Configures password validation rules.
"""

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

"""
=====================================
Internationalization
=====================================
Configures language and timezone settings.
"""

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

"""
=====================================
REST Framework & CORS
=====================================
Configures DRF authentication and CORS settings.
Updated CORS_ALLOWED_ORIGINS to include localhost origins for development
when DEBUG=True, fixing CORS configuration mismatch (Issue #4).
"""

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    )
}
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(hours=2),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
}
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = list(default_headers) + [
    "ngrok-skip-browser-warning",
]
CORS_ALLOWED_ORIGINS = [
    "https://intellivision.aionos.co",
    "http://34.100.200.148",
    "http://localhost:8080"
]
if DEBUG:
    CORS_ALLOWED_ORIGINS.extend([
        "http://localhost:3000",      # Development frontend (React/Vite)
        "http://127.0.0.1:3000",      # Alternative localhost
        "http://localhost:8080",
    ])

"""
=====================================
Celery
=====================================
Configures Celery for task processing, optimized for Tesla P100 GPU, 6 vCPUs, 27GB RAM.
"""

CELERY_BROKER_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
CELERY_WORKER_PREFETCH_MULTIPLIER = 2  # Reduced for GPU memory management
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_MAX_TASKS_PER_CHILD = 50  # Lower for GPU memory cleanup
CELERY_WORKER_MAX_MEMORY_PER_CHILD = 5000000  # 5GB per worker (3 workers * 5GB = 15GB)
CELERY_TASK_TIME_LIMIT = 7200  # 2 hours for complex GPU processing
CELERY_TASK_SOFT_TIME_LIMIT = 6600  # 110 minutes soft limit
CELERY_WORKER_CONCURRENCY = 3  # 3 concurrent jobs for 3 vCPUs
CELERY_TASK_ALWAYS_EAGER = False
CELERY_WORKER_DISABLE_RATE_LIMITS = True
CELERY_TASK_ROUTES = {
    'apps.video_analytics.tasks.*': {'queue': 'gpu_queue'},  # Route GPU tasks to dedicated queue
}
CELERY_WORKER_HIJACK_ROOT_LOGGER = False
CELERY_WORKER_LOG_FORMAT = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_SEND_SENT_EVENT = True

"""
=====================================
Logging
=====================================
Configures comprehensive logging for Docker environment with multiple handlers,
structured formatting, and separate log files for different components.
"""

# Use local logs directory for development, Docker path for production
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production').lower()
IS_LOCAL = ENVIRONMENT == 'local'

if IS_LOCAL:
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
else:
    LOG_DIR = '/app/intellivision/logs'

os.makedirs(LOG_DIR, exist_ok=True)

# Create subdirectories for organized logging
CELERY_LOG_DIR = os.path.join(LOG_DIR, 'celery')
API_LOG_DIR = os.path.join(LOG_DIR, 'api')
SECURITY_LOG_DIR = os.path.join(LOG_DIR, 'security')
PERFORMANCE_LOG_DIR = os.path.join(LOG_DIR, 'performance')

for log_subdir in [CELERY_LOG_DIR, API_LOG_DIR, SECURITY_LOG_DIR, PERFORMANCE_LOG_DIR]:
    os.makedirs(log_subdir, exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{asctime}] {levelname} {name} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
        'json': {
            '()': JSONFormatter,
        },
        'celery': {
            'format': '[{asctime}] {levelname} {processName} [{name}] {funcName}:{lineno} {message}',
            'style': '{',
        },
        'security': {
            'format': '[{asctime}] SECURITY {levelname} {name} | {message}',
            'style': '{',
        },
        'performance': {
            'format': '[{asctime}] PERF {levelname} {name} | {message}',
            'style': '{',
        },
    },
    'filters': {
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse',
        },
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'level': 'INFO',  # Show important logs
        },
        'console_debug': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'level': 'DEBUG',
            'filters': ['require_debug_true'],
        },
        'console_video': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'level': 'INFO',
        },
        'file_general': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'django.log'),
            'maxBytes': 50 * 1024 * 1024,  # 50MB
            'backupCount': 5,
            'formatter': 'verbose',
            'level': 'INFO',
        },
        'file_error': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'errors.log'),
            'maxBytes': 100 * 1024 * 1024,  # 100MB
            'backupCount': 10,
            'formatter': 'json',
            'level': 'ERROR',
        },
        'file_celery': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(CELERY_LOG_DIR, 'celery.log'),
            'maxBytes': 100 * 1024 * 1024,  # 100MB
            'backupCount': 5,
            'formatter': 'celery',
            'level': 'INFO',
        },
        'file_celery_error': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(CELERY_LOG_DIR, 'celery_errors.log'),
            'maxBytes': 50 * 1024 * 1024,  # 50MB
            'backupCount': 5,
            'formatter': 'json',
            'level': 'ERROR',
        },
        'file_api': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(API_LOG_DIR, 'api.log'),
            'maxBytes': 100 * 1024 * 1024,  # 100MB
            'backupCount': 7,
            'formatter': 'json',
            'level': 'INFO',
        },
        'file_security': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(SECURITY_LOG_DIR, 'security.log'),
            'maxBytes': 50 * 1024 * 1024,  # 50MB
            'backupCount': 10,
            'formatter': 'security',
            'level': 'INFO',
        },
        'file_performance': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(PERFORMANCE_LOG_DIR, 'performance.log'),
            'maxBytes': 50 * 1024 * 1024,  # 50MB
            'backupCount': 5,
            'formatter': 'performance',
            'level': 'INFO',
        },
    },
    'root': {
        'handlers': ['console', 'file_general', 'file_error'],
        'level': 'WARNING',
    },
    'loggers': {
        'django': {
            'handlers': ['file_general'],
            'level': 'WARNING',
            'propagate': False,
        },
        'django.request': {
            'handlers': ['file_api', 'file_error'],
            'level': 'WARNING',
            'propagate': False,
        },
        'django.security': {
            'handlers': ['file_security', 'file_error'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.db.backends': {
            'handlers': ['console_debug'],
            'level': 'WARNING',
            'propagate': False,
        },
        'celery': {
            'handlers': ['file_celery'],
            'level': 'WARNING',  # Reduce celery noise
            'propagate': False,
        },
        'celery.task': {
            'handlers': ['file_celery', 'file_celery_error'],
            'level': 'WARNING',  # Reduce task noise
            'propagate': False,
        },
        'apps.video_analytics': {
            'handlers': ['console_video', 'file_celery', 'file_performance'],
            'level': 'INFO',  # Keep video analytics visible
            'propagate': False,
        },
        'apps.video_analytics.analytics': {
            'handlers': ['console_video', 'file_celery', 'file_performance'],
            'level': 'INFO',  # Keep analytics visible
            'propagate': False,
        },
        'apps.video_analytics.tasks': {
            'handlers': ['console_video', 'file_celery', 'file_performance'],
            'level': 'INFO',  # Keep task progress visible
            'propagate': False,
        },
        'apps.face_auth': {
            'handlers': ['console', 'file_api', 'file_security'],
            'level': 'INFO',
            'propagate': False,
        },
        'security_logger': {
            'handlers': ['file_security', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        'performance_logger': {
            'handlers': ['file_performance'],
            'level': 'INFO',
            'propagate': False,
        },
        'api_logger': {
            'handlers': ['file_api', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

"""
=====================================
Third-Party Service Credentials
=====================================
Loads credentials from .env for external services.
"""

MONGO_URI = os.environ.get('MONGO_URI')
QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')

"""
=====================================
GPU and High-Performance Computing
=====================================
Optimizes for Tesla P100 GPU, 6 vCPUs, 27GB RAM.
"""

CUDA_VISIBLE_DEVICES = "0"  # Use the first (and only) GPU
TORCH_CUDA_ARCH_LIST = "6.0"  # Pascal architecture for Tesla P100
CUDA_CACHE_PATH = os.path.join(BASE_DIR, 'cuda_cache')
os.makedirs(CUDA_CACHE_PATH, exist_ok=True)

# Memory management for 27GB RAM
os.environ.setdefault('OMP_NUM_THREADS', '6')  # Match vCPU count
os.environ.setdefault('MKL_NUM_THREADS', '6')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '6')

# YOLO model cache optimization for SSD storage
YOLO_CACHE_DIR = os.environ.get('YOLO_CACHE_DIR', str(BASE_DIR / 'models' / 'yolo_cache'))
os.makedirs(YOLO_CACHE_DIR, exist_ok=True)

# Video processing optimization for Tesla P100
VIDEO_PROCESSING_BATCH_SIZE = 64  # Higher batch size for Tesla P100 (16GB VRAM)
MAX_VIDEO_RESOLUTION = (1920, 1080)  # Limit to prevent memory overflow
FRAME_SKIP_THRESHOLD = 1  # Process every frame for Tesla P100 performance
GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory

# Resource management
MAX_CONCURRENT_JOBS = 6  # Reduced from 10 to 6 for better memory management

# Memory management for web workers
import gc
gc.set_threshold(700, 10, 10)  # More aggressive garbage collection

# Rate limiting has been removed for better user experience

"""
=====================================
Cache and File Uploads
=====================================
Configures caching and file upload limits.
Confirms use of django.core.files.storage.FileSystemStorage for local storage.
"""

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
        'OPTIONS': {
            'MAX_ENTRIES': 10000,
            'CULL_FREQUENCY': 3,
        }
    }
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
FILE_UPLOAD_MAX_MEMORY_SIZE = 524288000  # 500MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 524288000  # 500MB

# =====================================
# Environment Variable Validation
# =====================================
def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        'DJANGO_SECRET_KEY',
        'POSTGRES_DB',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_HOST',
        'POSTGRES_PORT',
        'REDIS_URL',
        'QDRANT_URL',
        'MONGO_URI',
        'CLOUDINARY_CLOUD_NAME',
        'CLOUDINARY_API_KEY',
        'CLOUDINARY_API_SECRET',
        'AZURE_OPENAI_API_KEY',
        'ROBOFLOW_API_KEY',
    ]

    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate environment on startup
try:
    validate_environment()
except ValueError as e:
    import sys
    print(f"❌ Environment validation failed: {e}")
    print("Please check your .env file and ensure all required variables are set.")
    sys.exit(1)

# =====================================
# Database Configuration
# =====================================
