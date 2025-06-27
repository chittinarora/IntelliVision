"""
admin.py - Video Analytics App
Admin configuration for registering VideoJob model with the Django admin site.
"""

from django.contrib import admin
from .models import VideoJob

# Register the VideoJob model with the admin site
admin.site.register(VideoJob)
