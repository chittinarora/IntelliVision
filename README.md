# IntelliVision Django Project

This project provides a video analytics and face authentication platform using Django, Celery, and FastAPI. It supports people counting, car counting, emergency detection, food waste estimation, and more, with asynchronous processing and REST APIs.

## Setup Instructions

1. **Install all packages using requirements.txt**
   ```
   pip install -r requirements.txt
   ```
2. **Start the Django development server:**
   ```
   python intellivision/manage.py runserver
   ```
3. **Start the Celery worker for background tasks:**
   ```
   cd intellivision
   celery -A intellivision worker --pool=solo --loglevel=info
   ```
4. **(Optional) Expose your local server using Cloudflare Tunnel:**
   ```
   cloudflared tunnel --url http://localhost:8000
   ```

## Notes

- Ensure you have a `.env` file with the required environment variables in the project root.
- Media and output files are stored in the `media/` directory.
- Visit `/admin/` for the Django admin panel, `/api/jobs/` for job APIs, and `/api/faceauth/` for face authentication APIs.
