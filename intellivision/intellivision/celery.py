import multiprocessing as mp
import os
from celery import Celery

# Set the multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'intellivision.settings')

app = Celery('intellivision')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
