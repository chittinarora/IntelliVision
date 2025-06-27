"""
apps.py - Video Analytics App
App configuration for video analytics Django app.
"""

from django.apps import AppConfig


class TrackerConfig(AppConfig):
    """
    AppConfig for the video analytics app.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.video_analytics'
