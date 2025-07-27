# /apps/video_analytics/admin.py

from django.contrib import admin
from .models import VideoJob

"""
Admin configuration for the video analytics app, registering the VideoJob model
with custom display and filtering options.
"""

@admin.register(VideoJob)
class VideoJobAdmin(admin.ModelAdmin):
    """Admin interface for VideoJob model."""
    list_display = ('id', 'user', 'job_type', 'status', 'created_at', 'updated_at')
    list_filter = ('job_type', 'status', 'user')
    search_fields = ('user__username', 'job_type', 'task_id')
    readonly_fields = ('created_at', 'updated_at')