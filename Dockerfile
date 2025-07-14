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

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Default command (can be overridden in docker-compose)
CMD ["python", "intellivision/manage.py", "runserver", "0.0.0.0:8000"]
