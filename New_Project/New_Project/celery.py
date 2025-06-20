import multiprocessing as mp

mp.set_start_method('spawn', force=True)

import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'New_Project.settings')

app = Celery('New_Project')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
