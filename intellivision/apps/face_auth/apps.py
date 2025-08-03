"""
apps.py - Face Auth App
App configuration for face authentication Django app.
"""

from django.apps import AppConfig


class FaceauthConfig(AppConfig):
    """
    AppConfig for the face authentication app.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.face_auth'
    
    def ready(self):
        """Initialize face models when Django is ready (Celery workers only)"""
        try:
            from .embedding import preload_face_models
            preload_face_models()
        except Exception as e:
            # Silently ignore errors during startup to prevent app crashes
            pass
