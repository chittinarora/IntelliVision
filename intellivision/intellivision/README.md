python manage.py runserver

celery -A intellivision worker -l info

redis-server

cloudflared tunnel --url http://localhost:8000

source ../p_venv/bin/activate
