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
    ]

    JOB_TYPE_CHOICES = [
        ("people_count", "People Counting"),
        ("emergency_count", "Emergency Analysis"),
        ("car_count", "Car Counting"),
        ("pothole_detection", "Pothole Detection"),
        ("food_waste_estimation", "Food Waste Estimation"),
        ("pest_monitoring", "Pest Monitoring"),
        ("wildlife_detection", "Wildlife Detection")
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    input_video = models.FileField(upload_to='uploads/')
    output_video = models.FileField(upload_to='outputs/', null=True, blank=True)
    job_type = models.CharField(max_length=50, choices=JOB_TYPE_CHOICES, default="people_count")
    results = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    roi_x = models.FloatField(null=True, blank=True)
    roi_y = models.FloatField(null=True, blank=True)
    roi_width = models.FloatField(null=True, blank=True)
    roi_height = models.FloatField(null=True, blank=True)
