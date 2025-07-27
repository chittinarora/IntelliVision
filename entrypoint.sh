#!/bin/sh
set -e

# Run migrations
python intellivision/manage.py migrate

# Start Gunicorn on internal port 8001
exec gunicorn intellivision.wsgi:application --bind 0.0.0.0:8001 --workers 3