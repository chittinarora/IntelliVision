#!/bin/sh
# entrypoint.sh - Startup script for Django services (web and celery)
# Runs setup tasks (wait for PostgreSQL, conditional migrations) and executes the
# service-specific command (gunicorn for web, celery for celery).

set -e  # Exit immediately if any command fails

################################################################
# Section 1: Wait for PostgreSQL
# - Uses pg_isready to check if the intellivision database is ready.
# - Loops until the database is available, sleeping 2 seconds between attempts.
# - Uses string command (not array) to avoid sh error.
################################################################

echo "Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}..."
until pg_isready -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"; do
  echo "PostgreSQL not ready, retrying in 2 seconds..."
  sleep 2
done
echo "PostgreSQL is ready."

################################################################
# Section 2: Conditional Django Migrations
# - Runs migrations only if SERVICE_TYPE=web (set in docker-compose.yml).
# - Prevents celery from running migrations, avoiding duplicate table errors
#   (e.g., django_migrations_id_seq conflict).
# - Logs the action for debugging.
################################################################

if [ "$SERVICE_TYPE" = "web" ]; then
  echo "Running migrations for web service..."
  python intellivision/manage.py migrate
else
  echo "Skipping migrations for service: ${SERVICE_TYPE}"
fi

################################################################
# Section 3: Execute Service Command
# - Runs the command passed via docker-compose.yml (e.g., gunicorn for web, celery for celery).
# - Uses 'exec' to replace the shell process, ensuring proper signal handling
#   (e.g., SIGTERM for container shutdown).
# - Logs the command for debugging.
################################################################

echo "Starting service with command: $@"
exec "$@"