"""
models.py - Video Analytics App
Defines the VideoJob model for tracking video processing jobs and their results.
"""

from django.db import models
from django.contrib.auth.models import User


"""
Database models for the tracker app. Defines the VideoJob model and its fields.
"""


class VideoJob(models.Model):
    """
    Model representing a video processing job, including user, status, input/output videos, job type, results, and ROI.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('done', 'Done'),
        ('failed', 'Failed'),
        ('pending_config', 'Pending Configuration'),
    ]

    JOB_TYPE_CHOICES = [
        ("people_count", "People Counting"),
        ("emergency_count", "Emergency Analysis"),
        ("car_count", "Car Counting"),
        ("pothole_detection", "Pothole Detection"),
        ("food_waste_estimation", "Food Waste Estimation"),
        ("room_readiness", "Room Readiness Analysis"),
        ("wildlife_detection", "Wildlife Detection")
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, help_text="User who submitted the job.")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', help_text="Current status of the job.")
    input_video = models.FileField(upload_to='uploads/', help_text="Input video file for processing.")
    output_video = models.FileField(upload_to='outputs/', null=True, blank=True, help_text="Output video file after processing.")
    output_image = models.ImageField(upload_to='outputs/', null=True, blank=True, help_text="Output image file after processing (if applicable).")
    job_type = models.CharField(max_length=50, choices=JOB_TYPE_CHOICES, default="people_count", help_text="Type of analytics job.")
    results = models.JSONField(null=True, blank=True, help_text="Results of the analytics job.")
    created_at = models.DateTimeField(auto_now_add=True, help_text="Job creation timestamp.")
    updated_at = models.DateTimeField(auto_now=True, help_text="Job last updated timestamp.")
    # Line coordinates for emergency_count jobs (replaces ROI fields)
    emergency_lines = models.JSONField(null=True, blank=True, help_text="List of lines for emergency_count, each as dict with start/end x/y")
    video_width = models.IntegerField(null=True, blank=True, help_text="Width of the input video (optional)")
    video_height = models.IntegerField(null=True, blank=True, help_text="Height of the input video (optional)")
    lobby_zones = models.JSONField(null=True, blank=True, help_text="List of zones for lobby/crowd detection, each as dict with 'points' (list of [x, y]) and 'threshold' (int)")

    def __str__(self):
        return f"{self.job_type} (ID: {self.id})"
