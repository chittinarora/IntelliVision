# IntelliVision Django Project

This project provides a video analytics and face authentication platform using Django, Celery, and React. It supports people counting, car counting, emergency detection, food waste estimation, and more, with asynchronous processing and REST APIs.

## 🚀 Production Deployment

### System Requirements

- **CPU**: Intel Xeon @ 2.30GHz (6 vCPUs)
- **RAM**: 27GB
- **GPU**: Tesla P100-PCIE-16GB
- **Storage**: SSD recommended for model caching

### Apache Configuration

The system is configured to work with Apache as a reverse proxy. Apache configuration files are located at:

- `/etc/apache2/sites-available/intellivision.aionos.co.conf` (HTTP)
- `/etc/apache2/sites-available/intellivision.aionos.co-le-ssl.conf` (HTTPS)

**Note**: Docker containers now store media files in `./intellivision/media/` to match your existing Apache configuration. No Apache config changes needed!

#### Apache Setup Requirements:

```bash
# Enable required Apache modules
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod ssl
sudo a2enmod rewrite
sudo a2enmod headers
sudo a2enmod expires

# Enable the site
sudo a2ensite intellivision.aionos.co
sudo a2ensite intellivision.aionos.co-le-ssl

# Restart Apache
sudo systemctl restart apache2
```

#### Port Configuration:

- **Apache**: Ports 80 (HTTP) and 443 (HTTPS)
- **Django Backend**: Port 8001 (Gunicorn)
- **React Frontend**: Port 8080 (Vite)
- **PostgreSQL**: Port 5432
- **Redis**: Port 6379
- **Qdrant**: Port 6333

### Docker Deployment

1. **Build and start all services:**

   ```bash
   docker-compose up -d
   ```

   **Note**: Logs are automatically filtered to show only important events (video analytics, errors, system events)

2. **Check service status:**

   ```bash
   docker-compose ps
   docker-compose logs web
   docker-compose logs celery
   ```

3. **View filtered logs only:**

   ```bash
   ./show_important_logs.sh important
   ./show_important_logs.sh video follow
   ```

4. **Monitor GPU usage:**
   ```bash
   nvidia-smi
   docker exec -it intellivision-celery-1 nvidia-smi
   ```

## 🔧 Environment Configuration

### Required .env Variables:

```bash
# Django Configuration
DJANGO_SECRET_KEY=your-long-random-secret-key
DJANGO_DEBUG=False
ENVIRONMENT=production

# Database
POSTGRES_DB=intellivision
POSTGRES_USER=adminvision
POSTGRES_PASSWORD=IntelliVisionAIonOS
POSTGRES_HOST=db
POSTGRES_PORT=5432

# Redis
REDIS_URL=redis://redis:6379/0

# Analytics Processing Directories (Optional)
JOB_OUTPUT_DIR=/app/intellivision/media/outputs  # Default: /app/intellivision/media/outputs
JOB_TEMP_DIR=/app/intellivision/media/temp       # Default: /app/intellivision/media/temp

### Directory Configuration:

- **JOB_OUTPUT_DIR**: Directory where processed video outputs are stored
- **JOB_TEMP_DIR**: Directory for temporary files during processing (persistent across container restarts)
- Both directories are automatically created if they don't exist
- Default paths are within the media directory for better persistence

# GPU Configuration (Tesla P100)
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=6.0
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# External Services
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
AZURE_OPENAI_API_KEY=your-azure-openai-key
ROBOFLOW_API_KEY=your-roboflow-key

# Frontend Configuration
VITE_API_BASE_URL=https://intellivision.aionos.co/api
VITE_MEDIA_BASE_URL=https://intellivision.aionos.co/api/media
```

## 📊 Performance Optimization

### Tesla P100 GPU Configuration:

- **Batch Size**: 64 (optimized for 16GB VRAM)
- **Memory Fraction**: 80% (12.8GB usable)
- **Concurrency**: 3 workers (matching 3 vCPUs)
- **Memory per Worker**: 5GB (15GB total)

### Resource Allocation:

- **Web Service**: 1.5 vCPUs, 3GB RAM
- **Celery Workers**: 3 vCPUs, 16GB RAM
- **Database**: 0.5 vCPUs, 512MB RAM
- **Redis**: 0.5 vCPUs, 512MB RAM
- **Qdrant**: 0.2 vCPUs, 256MB RAM
- **Frontend**: 0.3 vCPUs, 512MB RAM

## 🔍 Monitoring & Logs

### Health Checks:

- **Backend**: `https://intellivision.aionos.co/health/`
- **Frontend**: `https://intellivision.aionos.co/`
- **Admin**: `https://intellivision.aionos.co/admin/`

### Log Locations:

- **Apache**: `/var/log/apache2/intellivision_*.log`
- **Django**: `/app/intellivision/logs/`
- **Celery**: `/app/intellivision/logs/celery/`

### Log Filtering:

The system automatically filters logs to show only important events:

- **Video Analytics**: Progress bars, job processing, GPU usage, FPS tracking
- **System Events**: Container start/stop, health checks, database migrations
- **Errors**: All error messages and exceptions
- **Security**: Authentication, login, permissions

**To view all logs (unfiltered):**

```bash
docker-compose logs -f
```

**To view specific filtered logs:**

```bash
./show_important_logs.sh video      # Video analytics only
./show_important_logs.sh system     # System events only
./show_important_logs.sh errors     # Errors only
./show_important_logs.sh important  # All important logs
```

- **Docker**: `docker-compose logs [service]`

## 🛠️ Development Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   cd intelli-vision && npm install
   ```

2. **Start development servers:**

   ```bash
   # Backend
   python intellivision/manage.py runserver

   # Frontend
   cd intelli-vision && npm run dev

   # Celery
   celery -A intellivision worker --loglevel=info
   ```

## 📁 File Structure

```
IntelliVision/
├── intellivision/          # Django backend
│   ├── apps/
│   │   ├── video_analytics/    # Video processing
│   │   └── face_auth/          # Face authentication
│   └── media/              # File storage
├── intelli-vision/         # React frontend
├── docker-compose.yml      # Production deployment
└── .env                    # Environment variables
```

## 🔐 Security Notes

- **SSL/TLS**: Configured via Let's Encrypt
- **File Uploads**: Limited to 100MB via Apache
- **Authentication**: JWT-based with face recognition
- **CORS**: Configured for production domain
- **Rate Limiting**: Disabled (removed for better user experience)

## 🚨 Troubleshooting

### Common Issues:

1. **GPU not detected**: Check NVIDIA Docker runtime
2. **Memory errors**: Reduce batch size or worker count
3. **Port conflicts**: Verify Apache and Docker ports
4. **SSL errors**: Check Let's Encrypt certificate renewal

### Debug Commands:

```bash
# Check GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check service status
docker-compose ps
docker-compose logs [service]

# Monitor resources
htop
nvidia-smi
df -h
```

## 📞 Support

- **API Documentation**: `https://intellivision.aionos.co/api/`
- **Admin Panel**: `https://intellivision.aionos.co/admin/`
- **Job Management**: `https://intellivision.aionos.co/dashboard/`

---

**Note**: This system is optimized for Tesla P100 GPU with 27GB RAM. Adjust resource allocation for different hardware configurations.
