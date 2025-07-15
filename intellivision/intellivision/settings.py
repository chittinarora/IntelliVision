# ======================================
# Django settings for intellivision
# ======================================

import os
from pathlib import Path
from dotenv import load_dotenv
from corsheaders.defaults import default_headers
from datetime import timedelta

# === Path & Environment Setup ===
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')  # Loads .env from project root

# === Environment Variables / Secrets ===
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-aq_x8cykh&3r_q9df@b%n(p(dv5&3gt$=17m#u-ir$kzl-(mjm')
DEBUG = False


SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(hours=2),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
}

# === Production Database Example (PostgreSQL) ===
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('POSTGRES_DB', 'intellivision'),
        'USER': os.environ.get('POSTGRES_USER', 'adminvision'),
        'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'IntelliVisionAIonOS'),
        'HOST': os.environ.get('POSTGRES_HOST', 'localhost'),
        'PORT': os.environ.get('POSTGRES_PORT', '5432'),
    }
}

# === Development Database (SQLite, not for production) ===
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#     }
# }

# === Password Validation ===
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# === Internationalization ===
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# === Static & Media Files ===
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')  # For collectstatic, serve with Nginx in production
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')         # Serve with Nginx in production

# === Security Settings for Production ===
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_REFERRER_POLICY = "strict-origin"
# SECURE_CONTENT_TYPE_NOSNIFF = True
# SECURE_BROWSER_XSS_FILTER = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# === ALLOWED_HOSTS ===
# For production, add your domain or public IP
ALLOWED_HOSTS = ['intellivision.aionos.co']

# === Third-Party Service Credentials ===
MONGO_URI = os.environ.get('MONGO_URI')
QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')

# === Unified Output Directory for Analytics Jobs ===
# All analytics jobs (people_count, car_count, anpr, etc.) should save outputs here.
# Set JOB_OUTPUT_DIR in your .env to override, or use the default below.
JOB_OUTPUT_DIR = os.environ.get('JOB_OUTPUT_DIR', str(BASE_DIR / 'media' / 'outputs'))

# === Application Definition ===
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

ROOT_URLCONF = 'intellivision.urls'

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

WSGI_APPLICATION = 'intellivision.wsgi.application'

# === Celery Configuration ===
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_ACKS_LATE = True

# Additional settings to prevent SIGSEGV and improve stability
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1  # Restart worker after each task
CELERY_WORKER_MAX_MEMORY_PER_CHILD = 200000  # 200MB memory limit per worker
CELERY_TASK_TIME_LIMIT = 3600  # 1 hour time limit
CELERY_TASK_SOFT_TIME_LIMIT = 3000  # 50 minutes soft limit
CELERY_WORKER_CONCURRENCY = 1  # Use single worker to avoid conflicts
CELERY_TASK_ALWAYS_EAGER = False  # Ensure tasks run in background
CELERY_WORKER_DISABLE_RATE_LIMITS = True  # Disable rate limiting for ML tasks

# === Django REST Framework ===
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    )
}

# === CORS Headers ===
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = list(default_headers) + [
    "ngrok-skip-browser-warning",
]
CORS_ALLOWED_ORIGINS = ["https://intellivision.aionos.co"]

# === Miscellaneous ===
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# === Logging Configuration ===
# Ensure logs directory exists for file-based logging
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
