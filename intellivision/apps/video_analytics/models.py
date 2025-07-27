# /apps/video_analytics/models.py

"""
=====================================
Imports
=====================================
Defines the VideoJob model for tracking video and image processing jobs.
Added jsonschema for results field validation (Issue #7).
"""

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
from jsonschema import validate, ValidationError as JSONSchemaValidationError

"""
=====================================
Results Schema Validation
=====================================
Defines JSON schemas for results field based on job types from job.ts.
Validates results structure to prevent inconsistent data (Issue #7).
"""

RESULT_SCHEMAS = {
    "people-count": {
        "type": "object",
        "required": ["processed_frames", "total_frames", "in_count", "out_count", "current_count"],
        "properties": {
            "processed_frames": {"type": "number"},
            "total_frames": {"type": "number"},
            "in_count": {"type": "number"},
            "out_count": {"type": "number"},
            "current_count": {"type": "number"}
        }
    },
    "car-count": {
        "type": "object",
        "required": ["processed_frames", "total_frames", "car_count"],
        "properties": {
            "processed_frames": {"type": "number"},
            "total_frames": {"type": "number"},
            "car_count": {"type": "number"}
        }
    },
    "emergency-count": {
        "type": "object",
        "required": ["processed_frames", "total_frames", "in_count", "out_count", "current_count"],
        "properties": {
            "processed_frames": {"type": "number"},
            "total_frames": {"type": "number"},
            "in_count": {"type": "number"},
            "out_count": {"type": "number"},
            "current_count": {"type": "number"}
        }
    },
    "pothole-detection": {
        "type": "object",
        "required": ["processed_frames", "total_frames", "pothole_count", "potholes"],
        "properties": {
            "processed_frames": {"type": "number"},
            "total_frames": {"type": "number"},
            "pothole_count": {"type": "number"},
            "potholes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["pothole_id", "bounding_box", "severity", "confidence"],
                    "properties": {
                        "pothole_id": {"type": "number"},
                        "bounding_box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "confidence": {"type": "number"}
                    }
                }
            }
        }
    },
    "food-waste-estimation": {
        "type": "object",
        "required": ["processed_frames", "total_frames"],
        "properties": {
            "processed_frames": {"type": "number"},
            "total_frames": {"type": "number"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "estimated_portion", "estimated_calories", "tags"],
                    "properties": {
                        "name": {"type": "string"},
                        "estimated_portion": {"type": ["string", "number"]},
                        "estimated_calories": {"type": "number"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "total_calories": {"type": "number"},
            "waste_summary": {"type": "string"}
        }
    },
    "room-readiness": {
        "type": "object",
        "required": ["processed_frames", "total_frames", "is_ready", "checklist", "issues"],
        "properties": {
            "processed_frames": {"type": "number"},
            "total_frames": {"type": "number"},
            "is_ready": {"type": "boolean"},
            "checklist": {
                "type": "object",
                "required": ["bed_made", "trash_empty", "surfaces_clean", "floor_clean", "no_items_left"],
                "properties": {
                    "bed_made": {"type": "boolean"},
                    "trash_empty": {"type": "boolean"},
                    "surfaces_clean": {"type": "boolean"},
                    "floor_clean": {"type": "boolean"},
                    "no_items_left": {"type": "boolean"}
                },
                "additionalProperties": {"type": "boolean"}
            },
            "issues": {"type": "array", "items": {"type": "string"}}
        }
    },
    "wildlife-detection": {
        "type": "object",
        "required": ["processed_frames", "total_frames"],
        "properties": {
            "processed_frames": {"type": "number"},
            "total_frames": {"type": "number"},
            "wildlife_detected": {"type": "boolean"},
            "wildlife_count": {"type": "number"},
            "wildlife_types": {"type": "array", "items": {"type": "string"}},
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]}
        }
    },
    "lobby-detection": {
        "type": "object",
        "required": ["processed_frames", "total_frames", "overall_people_count", "crowd_density", "average_wait_time_seconds", "zones"],
        "properties": {
            "processed_frames": {"type": "number"},
            "total_frames": {"type": "number"},
            "overall_people_count": {"type": "number"},
            "crowd_density": {"type": "number"},
            "average_wait_time_seconds": {"type": "number"},
            "zones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["zone_id", "people_count"],
                    "properties": {
                        "zone_id": {"type": "string"},
                        "people_count": {"type": "number"},
                        "queue_length": {"type": "number"}
                    }
                }
            }
        }
    },
    "parking-analysis": {
        "type": "object",
        "required": ["processed_frames", "total_frames"],
        "properties": {
            "processed_frames": {"type": "number"},
            "total_frames": {"type": "number"},
            "occupied_spaces": {"type": "number"},
            "total_spaces": {"type": "number"},
            "availability": {"type": "number"},
            "zones_analyzed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["zone", "occupied", "total"],
                    "properties": {
                        "zone": {"type": "string"},
                        "occupied": {"type": "number"},
                        "total": {"type": "number"}
                    }
                }
            }
        }
    }
}

def validate_results_schema(value):
    """
    Validate the results JSONField against the schema for the job_type.
    Raises ValidationError if the schema is invalid or job_type is unknown.
    """
    if value is None:
        return value
    instance = getattr(validate_results_schema, 'instance', None)
    if not instance or not hasattr(instance, 'job_type'):
        return value  # Skip validation if job_type is not accessible
    job_type = instance.job_type
    schema = RESULT_SCHEMAS.get(job_type)
    if not schema:
        raise models.ValidationError(f"No schema defined for job_type: {job_type}")
    try:
        validate(instance=value, schema=schema)
    except JSONSchemaValidationError as e:
        raise models.ValidationError(f"Invalid results format for {job_type}: {str(e)}")
    return value

"""
=====================================
VideoJob Model
=====================================
Represents a video or image processing job submitted by a user.
Tracks job status, input/output files, and results.
Added validator to results field to enforce job-type-specific schemas (Issue #7).
"""

class VideoJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    JOB_TYPE_CHOICES = [
        ("people-count", "People Counting"),
        ("emergency-count", "Emergency Analysis"),
        ("car-count", "Car Counting / ANPR"),
        ("parking-analysis", "Parking Lot Analysis"),
        ("pothole-detection", "Pothole Detection"),
        ("food-waste-estimation", "Food Waste Estimation"),
        ("room-readiness", "Room Readiness Analysis"),
        ("wildlife-detection", "Wildlife Detection"),
        ("lobby-detection", "Lobby / Crowd Detection"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.CASCADE,
        help_text="User who submitted the job."
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default='pending',
        help_text="Current status of the job."
    )
    job_type = models.CharField(
        max_length=50, choices=JOB_TYPE_CHOICES, default="people-count",
        help_text="Type of analytics to perform."
    )
    results = models.JSONField(
        null=True, blank=True,
        validators=[validate_results_schema],
        help_text="JSON results from analytics job, validated against job-type-specific schema."
    )
    input_video = models.FileField(
        upload_to='uploads/',
        validators=[FileExtensionValidator(['mp4', 'jpg', 'jpeg', 'png'])],
        help_text="Input video or image file."
    )
    output_video = models.FileField(
        upload_to='outputs/', null=True, blank=True,
        validators=[FileExtensionValidator(['mp4', 'webm', 'mov'])],
        help_text="Output video file."
    )
    output_image = models.ImageField(
        upload_to='outputs/', null=True, blank=True,
        validators=[FileExtensionValidator(['jpg', 'jpeg', 'png'])],
        help_text="Output image file."
    )
    youtube_url = models.URLField(
        max_length=512, null=True, blank=True,
        help_text="Source YouTube URL, if provided."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when job was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when job was last updated."
    )
    task_id = models.CharField(
        max_length=255, null=True, blank=True,
        help_text="Celery task ID for job."
    )
    emergency_lines = models.JSONField(
        null=True, blank=True,
        help_text="Line definitions for emergency_count."
    )
    lobby_zones = models.JSONField(
        null=True, blank=True,
        help_text="Zone definitions for lobby_detection."
    )
    video_width = models.IntegerField(
        null=True, blank=True,
        help_text="Input video width."
    )
    video_height = models.IntegerField(
        null=True, blank=True,
        help_text="Input video height."
    )

    def __str__(self) -> str:
        """String representation of VideoJob."""
        return f"Job {self.id}: {self.get_job_type_display()} for {self.user.username} ({self.status})"

    def save(self, *args, **kwargs):
        """Override save to provide instance context for results validation."""
        validate_results_schema.instance = self
        super().save(*args, **kwargs)
        validate_results_schema.instance = None