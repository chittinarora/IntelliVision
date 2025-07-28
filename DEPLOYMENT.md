# IntelliVision Deployment Guide

## Overview

This project uses Apache2 as a reverse proxy with Docker containers for the backend services.

## Architecture

- **Apache2**: Reverse proxy running on the host
  - `/api/` and `/api/media/` → Django backend (localhost:8001)
  - Everything else → React frontend (localhost:8080)
- **Django Backend**: Docker container on port 8001
- **React Frontend**: Development server on port 8080
- **Database**: PostgreSQL container
- **Cache**: Redis container
- **Vector DB**: Qdrant container

## Apache2 Configuration

Your Apache2 configuration is already set up at:

- `/etc/apache2/sites-available/intellivision.aionos.co.conf` (HTTP)
- `/etc/apache2/sites-available/intellivision.aionos.co-le-ssl.conf` (HTTPS)

## Deployment Steps

### 1. Start Backend Services

```bash
# Start all backend services (Django, PostgreSQL, Redis, Qdrant)
docker-compose up -d

# Check if services are running
docker-compose ps
```

### 2. Start Frontend Development Server

```bash
# Navigate to frontend directory
cd intelli-vision-old

# Install dependencies (if not already done)
npm install

# Start development server
npm run dev
```

### 3. Verify Services

- Django Backend: http://localhost:8001
- React Frontend: http://localhost:8080
- Apache2 Proxy: https://intellivision.aionos.co

### 4. Collect Static Files (if needed)

```bash
# Run inside the Django container
docker-compose exec web python intellivision/manage.py collectstatic --noinput
```

## Environment Variables

Make sure your `.env` file contains:

```env
DJANGO_SECRET_KEY=your-secret-key
DJANGO_DEBUG=False
POSTGRES_DB=intellivision
POSTGRES_USER=adminvision
POSTGRES_PASSWORD=IntelliVisionAIonOS
POSTGRES_HOST=db
POSTGRES_PORT=5432
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333
MONGO_URI=your-mongo-uri
```

## Troubleshooting

### Apache2 Issues

```bash
# Check Apache2 configuration
sudo apache2ctl configtest

# Restart Apache2
sudo systemctl restart apache2

# Check Apache2 logs
sudo tail -f /var/log/apache2/intellivision_error.log
```

### Docker Issues

```bash
# Check container logs
docker-compose logs web

# Restart services
docker-compose restart

# Rebuild containers
docker-compose up --build
```

### Frontend Issues

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check if port 8080 is available
lsof -i :8080
```

## File Structure

```
IntelliVision-1/
├── intellivision/           # Django backend
│   ├── intellivision/
│   │   ├── settings.py      # Updated for Apache2 proxy
│   │   └── urls.py
│   └── media/              # Media files (mounted in Docker)
├── intelli-vision-old/      # React frontend
│   ├── vite.config.ts      # Updated for proxy
│   └── package.json
├── docker-compose.yml       # Backend services
└── Dockerfile              # Django container
```

## Notes

- The Django backend runs on port 8001 inside Docker
- Apache2 proxies requests to localhost:8001
- Media files are served directly by Apache2 from `/home/aditya_dubey/IntelliVision/intellivision/media/`
- Static files are collected to `intellivision/staticfiles/` and served by Django
- CORS is configured to allow requests from the Apache2 proxy
