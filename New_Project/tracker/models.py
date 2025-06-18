from django.db import models
from django.contrib.auth.models import User


class VideoJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('done', 'Done'),
        ('failed', 'Failed'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    input_video = models.FileField(upload_to='uploads/')
    output_video = models.FileField(upload_to='outputs/', null=True, blank=True)
    job_type = models.CharField(max_length=50, default="people_count")
    person_count = models.IntegerField(null=True, blank=True)
    car_count = models.IntegerField(null=True, blank=True)
    in_count = models.IntegerField(null=True, blank=True)
    out_count = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)