# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential libpq-dev gcc cmake git qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools && \
    apt-get clean

# Install system dependencies
RUN apt-get update && apt-get install -y tesseract-ocr

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Ensure logs directory exists for Django logging
RUN mkdir -p intellivision/logs

# Collect static files
RUN python intellivision/manage.py collectstatic --noinput

# Run migrations and start Gunicorn
CMD ["sh", "-c", "python intellivision/manage.py migrate && gunicorn intellivision.wsgi:application --bind 0.0.0.0:8000 --workers 3"]
