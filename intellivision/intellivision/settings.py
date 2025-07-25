# ======================================
# Django settings for intellivision
# ======================================

# 1. Path & Environment
import os
from pathlib import Path
from dotenv import load_dotenv
from corsheaders.defaults import default_headers
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')  # Loads .env from project root

# 2. Security
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-aq_x8cykh&3r_q9df@b%n(p(dv5&3gt$=17m#u-ir$kzl-(mjm')
DEBUG = os.environ.get("DJANGO_DEBUG", "False") == "True"

# Only redirect to HTTPS in production
if not DEBUG:
    SECURE_SSL_REDIRECT = True
else:
    SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_REFERRER_POLICY = "strict-origin"
# SECURE_CONTENT_TYPE_NOSNIFF = True
# SECURE_BROWSER_XSS_FILTER = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
ALLOWED_HOSTS = ['intellivision.aionos.co', '34.100.200.148', '35.190.199.96', 'localhost', '127.0.0.1']

# 3. Database - Optimized for 27GB RAM
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('POSTGRES_DB', 'intellivision'),
        'USER': os.environ.get('POSTGRES_USER', 'adminvision'),
        'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'IntelliVisionAIonOS'),
        'HOST': os.environ.get('POSTGRES_HOST', 'localhost'),
        'PORT': os.environ.get('POSTGRES_PORT', '5432'),
        'OPTIONS': {
            'MAX_CONNS': 50,  # Higher connection pool for concurrent processing
        },
        'CONN_MAX_AGE': 600,  # 10 minutes connection reuse
    }
}
# For development, you can use SQLite by uncommenting below:
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#     }
# }

# 4. Installed Apps
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

# 5. Middleware
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

# 6. Templates
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

# 7. WSGI/ASGI
ROOT_URLCONF = 'intellivision.urls'
WSGI_APPLICATION = 'intellivision.wsgi.application'

# 8. Static & Media Files
STATIC_URL = '/api/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')  # For collectstatic, serve with Nginx in production
MEDIA_URL = '/api/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')         # Serve with Nginx in production

# 9. Authentication & Password Validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# 10. Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# 11. REST Framework & CORS
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
CORS_ALLOWED_ORIGINS = ["https://intellivision.aionos.co", "http://34.100.200.148"]

# 12. Celery - Optimized for Tesla P100 GPU + 6 vCPU + 27GB RAM
CELERY_BROKER_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # Reduced for GPU memory management
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_MAX_TASKS_PER_CHILD = 50  # Lower for GPU memory cleanup
CELERY_WORKER_MAX_MEMORY_PER_CHILD = 2000000  # 2GB per worker (higher for GPU tasks)
CELERY_TASK_TIME_LIMIT = 7200  # 2 hours for complex GPU processing
CELERY_TASK_SOFT_TIME_LIMIT = 6600  # 110 minutes soft limit
CELERY_WORKER_CONCURRENCY = 2  # Reduced to allow GPU memory per task (Tesla P100 has 16GB)
CELERY_TASK_ALWAYS_EAGER = False  # Ensure tasks run in background
CELERY_WORKER_DISABLE_RATE_LIMITS = True  # Disable rate limiting for ML tasks
CELERY_TASK_ROUTES = {
    'apps.video_analytics.tasks.*': {'queue': 'gpu_queue'},  # Route GPU tasks to dedicated queue
}
CELERY_WORKER_HIJACK_ROOT_LOGGER = False
CELERY_WORKER_LOG_FORMAT = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'

# 13. Logging
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
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
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'django.log'),
            'formatter': 'verbose',
            'level': 'WARNING',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.request': {
            'handlers': ['console', 'file'],
            'level': 'ERROR',
            'propagate': False,
        },
    },
}

# 14. Third-Party Service Credentials
MONGO_URI = os.environ.get('MONGO_URI')
QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')

# 15. Analytics Output Directory
# All analytics jobs (people_count, car_count, anpr, etc.) should save outputs here.
# Set JOB_OUTPUT_DIR in your .env to override, or use the default below.
JOB_OUTPUT_DIR = os.environ.get('JOB_OUTPUT_DIR', str(BASE_DIR / 'media' / 'outputs'))

# 16. GPU and High-Performance Computing Settings
# Tesla P100 GPU with CUDA 12.2 optimizations
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

# Video processing optimization
VIDEO_PROCESSING_BATCH_SIZE = 32  # Higher batch size for Tesla P100
MAX_VIDEO_RESOLUTION = (1920, 1080)  # Limit to prevent memory overflow
FRAME_SKIP_THRESHOLD = 2  # Process every 2nd frame for performance

# 17. Miscellaneous
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# 18. Session and Cache Settings - Optimized for high-performance
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL', 'redis://redis:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 100,  # Higher for concurrent processing
            }
        }
    }
}

# === .env Usage ===
# Place all secrets and sensitive config in your .env file (never commit to git):
# DJANGO_SECRET_KEY=your-secret-key
# DJANGO_DEBUG=False
# POSTGRES_DB=your_db
# POSTGRES_USER=your_user
# POSTGRES_PASSWORD=your_password
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# ... other secrets ...

# === Static/Media File Serving (Nginx Example) ===
# In production, serve /static/ and /media/ with Nginx, not Django.
# Example Nginx config:
# location /static/ { alias /path/to/staticfiles/; }
# location /media/  { alias /path/to/media/; }
