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
