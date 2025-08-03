# Face auth package
# Prevent heavy imports during Django startup to avoid memory corruption
import sys
import os

# Skip face auth initialization during Django management commands
if any(cmd in sys.argv for cmd in ['migrate', 'collectstatic', 'showmigrations', 'makemigrations']):
    # Disable face auth app loading during startup
    pass
else:
    # Normal face auth initialization only when actually running the app
    pass