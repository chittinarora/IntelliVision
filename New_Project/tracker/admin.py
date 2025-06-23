from django.contrib import admin

"""
Admin configuration for the tracker app. Registers VideoJob model with the admin site.
"""

# Register your models here.
from django.contrib import admin
from .models import VideoJob

# Register the VideoJob model with the admin site
admin.site.register(VideoJob)
