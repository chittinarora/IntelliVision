# /intellivision/intellivision/celery.py

"""
=====================================
Imports
=====================================
Configures Celery for the IntelliVision project.
"""

import os
import platform
import multiprocessing as mp
import logging

"""
=====================================
Logging Setup
=====================================
Sets up logging for debugging multiprocessing configuration.
"""

logger = logging.getLogger(__name__)

"""
=====================================
Multiprocessing Setup
=====================================
Sets multiprocessing start method based on platform for compatibility and performance.
Replaced hardcoded 'spawn' with platform-specific logic (Issue #9).
Prefers 'fork' on Linux for efficiency, 'spawn' on Windows/macOS.
"""

try:
    if platform.system() == 'Linux':
        mp.set_start_method('fork', force=True)  # Faster for Linux (VM)
        logger.info("Multiprocessing start method set to 'fork'")
    else:
        mp.set_start_method('spawn', force=True)  # Required for Windows/macOS
        logger.info("Multiprocessing start method set to 'spawn'")
except (ValueError, RuntimeError) as e:
    logger.warning(f"Failed to set multiprocessing start method: {e}")

"""
=====================================
Celery App
=====================================
Initializes the Celery application.
"""

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'intellivision.settings')

app = Celery('intellivision')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """Debug task for inspecting Celery configuration."""
    logger.debug(f'Request: {self.request!r}')