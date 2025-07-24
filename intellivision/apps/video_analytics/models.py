"""
models.py - Video Analytics App
Defines the VideoJob model for tracking video processing jobs and their results.
"""

from django.db import models
from django.contrib.auth.models import User


class VideoJob(models.Model):
    """
    Represents a single video or image processing job submitted by a user.
    This model tracks the job's status, input/output files, and final results.
    """
    # Choices for the job's current processing state.
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('done', 'Done'),
        ('failed', 'Failed'),
    ]

    # Choices for the type of analytics to be performed.
    # This list is aligned with the processors defined in tasks.py.
    JOB_TYPE_CHOICES = [
        ("people_count", "People Counting"),
        ("emergency_count", "Emergency Analysis"),
        ("car_count", "Car Counting / ANPR"),
        ("parking_analysis", "Parking Lot Analysis"),
        ("pothole_detection", "Pothole Detection"),
        ("food_waste_estimation", "Food Waste Estimation"),
        ("room_readiness", "Room Readiness Analysis"),
        ("wildlife_detection", "Wildlife Detection"),
        ("lobby_detection", "Lobby / Crowd Detection"),
    ]

    # --- Core Job Fields ---
    user = models.ForeignKey(User, on_delete=models.CASCADE, help_text="The user who submitted the job.")
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        help_text="The current status of the job (e.g., pending, processing)."
    )
    job_type = models.CharField(
        max_length=50,
        choices=JOB_TYPE_CHOICES,
        default="people_count",
        help_text="The type of analytics to be performed on the input file."
    )
    results = models.JSONField(null=True, blank=True, help_text="The JSON results returned by the analytics job.")

    # --- File Fields ---
    input_video = models.FileField(upload_to='uploads/', help_text="The input video or image file for processing.")
    output_video = models.FileField(upload_to='outputs/', null=True, blank=True,
                                    help_text="The resulting output video file after processing.")
    output_image = models.ImageField(upload_to='outputs/', null=True, blank=True,
                                     help_text="The resulting output image file after processing.")
    youtube_url = models.URLField(
        max_length=512, null=True, blank=True,
        help_text="The source YouTube URL, if provided by the user."
    )

    # --- Timestamps ---
    created_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp when the job was created.")
    updated_at = models.DateTimeField(auto_now=True, help_text="Timestamp when the job was last updated.")

    # --- Job-Specific Parameters ---
    emergency_lines = models.JSONField(
        null=True, blank=True,
        help_text="For 'emergency_count': A list of line definitions, each with start/end coordinates and direction."
    )
    lobby_zones = models.JSONField(
        null=True, blank=True,
        help_text="For 'lobby_detection': A list of zone definitions, each with points and a crowd threshold."
    )
    video_width = models.IntegerField(null=True, blank=True,
                                      help_text="The width of the input video, used for scaling coordinates.")
    video_height = models.IntegerField(null=True, blank=True,
                                       help_text="The height of the input video, used for scaling coordinates.")

    def __str__(self):
        """String representation of the VideoJob model."""
        return f"Job {self.id}: {self.get_job_type_display()} for {self.user.username} ({self.status})"
