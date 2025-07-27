# /apps/video_analytics/apps.py

from django.apps import AppConfig

"""
App configuration for the video analytics Django app.
"""

class TrackerConfig(AppConfig):
    """Configuration for the video analytics app."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.video_analytics'
    verbose_name = 'Video Analytics'