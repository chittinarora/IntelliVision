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
import inspect
import logging

logger = logging.getLogger(__name__)

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
        "required": ["person_count"],
        "properties": {
            "person_count": {"type": "number"},
            "raw_track_count": {"type": "number"},
            "depth_method": {"type": "string"},
            "fps": {"type": ["number", "null"]},
            "detection_strategies": {
                "type": "array",
                "items": {"type": "string"}
            },
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        }
    },
    "car-count": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "object",
                "properties": {
                    "entries": {"type": "number"},
                    "exits": {"type": "number"},
                    "max_occupancy": {"type": "number"},
                    "final_occupancy": {"type": "number"},
                    "recognized_plates": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "locked_plates": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "candidate_plates": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "processing_fps": {"type": "number"},
                    "total_frames": {"type": "number"},
                    "processing_time": {"type": "number"},
                    "vehicle_count": {"type": "number"},
                    "detection_stats": {
                        "type": "object",
                        "properties": {
                            "locked_count": {"type": "number"},
                            "tracked_vehicles": {"type": "number"},
                            "total_unique_vehicles": {"type": "number"}
                        }
                    }
                }
            },
            "preview_url": {"type": "string"},
            "download_url": {"type": "string"},
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "plate": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                }
            },
            "annotated_image": {"type": "string"},
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        }
    },
    "emergency-count": {
        "type": "object",
        "required": ["in_count", "out_count", "fast_in_count", "fast_out_count", "alerts"],
        "properties": {
            "in_count": {"type": "number"},
            "out_count": {"type": "number"},
            "fast_in_count": {"type": "number"},
            "fast_out_count": {"type": "number"},
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        }
    },
    "food-waste-estimation": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "estimated_portion_grams": {"type": "number"},
                        "estimated_calories": {"type": "number"},
                        "percent_uneaten": {"type": "number"},
                        "confidence_score": {"type": "number"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["name", "estimated_portion_grams", "estimated_calories", "percent_uneaten", "confidence_score"]
                }
            },
            "total_calories_served": {"type": "number"},
            "total_calories_wasted": {"type": "number"},
            "overall_waste_percentage": {"type": "number"},
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "timestamp": {"type": "string"}
                    },
                    "required": ["message", "timestamp"]
                }
            }
        }
    },
    "room-readiness": {
        "type": "object",
        "required": ["readiness_score", "total_issues"],
        "properties": {
            "readiness_score": {"type": "number"},
            "status": {"type": "string"},
            "zone": {"type": "string"},
            "checklist": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "parameter": {"type": "string"},
                        "status": {"type": "string"},
                        "box": {
                            "oneOf": [
                                {"type": "null"},
                                {"type": "array", "items": {"type": "number"}}
                            ]
                        },
                        "notes": {"type": "string"},
                        "ai_suggested_fix": {"type": "string"}
                    }
                }
            },
            "fail_reasons": {
                "type": "array",
                "items": {"type": "string"}
            },
            "instructions": {
                "type": "array",
                "items": {"type": "string"}
            },
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "message": {"type": "string"},
                        "severity": {"type": "string"}
                    }
                }
            },
            "image_width": {"type": "number"},
            "image_height": {"type": "number"},
            "overall_status": {"type": "string"},
            "rooms_analyzed": {"type": "number"},
            "rooms_ready": {"type": "number"},
            "total_issues": {"type": "number"},
            "rooms": {"type": "object"},
            "all_issues": {"type": "array"},
            "individual_reports": {"type": "array"},
            "deduplication_stats": {
                "type": "object",
                "properties": {
                    "raw_issues": {"type": "number"},
                    "unique_issues": {"type": "number"},
                    "efficiency": {"type": "string"}
                }
            }
        }
    },
    "pothole-detection": {
        "type": "object",
        "required": ["total_potholes"],
        "properties": {
            "total_potholes": {"type": "number"},
            "potholes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "confidence": {"type": "number"},
                        "class": {"type": "string"}
                    }
                }
            },
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame_index": {"type": "number"},
                        "potholes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "width": {"type": "number"},
                                    "height": {"type": "number"},
                                    "confidence": {"type": "number"},
                                    "class": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            },
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        }
    },
    "wildlife-detection": {
        "type": "object",
        "properties": {
            "detected_snakes": {"type": "number"},
            "detected_animals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                }
            },
            "mongo_id": {"type": "string"},
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        }
    },
    "lobby-detection": {
        "type": "object",
        "properties": {
            "zone_counts": {
                "type": "object",
                "additionalProperties": {"type": "number"}
            },
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "zone_name": {"type": "string"},
                        "message": {"type": "string"},
                        "count": {"type": "number"},
                        "threshold": {"type": "number"},
                        "video_timestamp": {"type": "string"},
                        "system_timestamp": {"type": "string"},
                        "frame_number": {"type": "number"}
                    }
                }
            },
            "alert_summary": {
                "type": "object",
                "properties": {
                    "total_alerts": {"type": "number"},
                    "alerts_by_zone": {
                        "type": "object",
                        "additionalProperties": {"type": "number"}
                    }
                }
            }
        }
    },
    "parking-analysis": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "object",
                "properties": {
                    "entries": {"type": "number"},
                    "exits": {"type": "number"},
                    "max_occupancy": {"type": "number"},
                    "final_occupancy": {"type": "number"},
                    "recognized_plates": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "locked_plates": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "candidate_plates": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "processing_fps": {"type": "number"},
                    "total_frames": {"type": "number"},
                    "processing_time": {"type": "number"},
                    "vehicle_count": {"type": "number"},
                    "detection_stats": {
                        "type": "object",
                        "properties": {
                            "locked_count": {"type": "number"},
                            "tracked_vehicles": {"type": "number"},
                            "total_unique_vehicles": {"type": "number"}
                        }
                    }
                }
            },
            "preview_url": {"type": "string"},
            "download_url": {"type": "string"},
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        }
    },
    "youtube_frame_extraction": {
        "type": "object",
        "required": ["frame_url"],
        "properties": {
            "frame_url": {"type": "string"},
            "method": {"type": "string"},
            "title": {"type": "string"},
            "file_size_mb": {"type": "number"},
            "processing_time": {"type": "number"}
        }
    }
}

def validate_results_schema(value):
    """
    Validate the results JSONField against the schema for the job_type.
    Raises ValidationError if the schema is invalid or job_type is unknown.

    Note: This validator is called during model save operations.
    The job_type is available through the model instance being validated.
    """
    if value is None:
        return value

    # Get the model instance from the validator context
    # This works because Django passes the model instance to validators
    # during model clean() and save() operations
    frame = inspect.currentframe()
    try:
        # Look up the call stack to find the model instance
        while frame:
            if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'job_type'):
                instance = frame.f_locals['self']
                break
            frame = frame.f_back
        else:
            # If we can't find the instance, skip validation
            # This happens during deserialization or other edge cases
            logger.warning("Results validation skipped: Could not determine job_type context")
            return value
    finally:
        del frame

    job_type = instance.job_type
    schema = RESULT_SCHEMAS.get(job_type)

    if not schema:
        logger.warning(f"No schema defined for job_type: {job_type}")
        return value  # Don't fail validation for missing schemas

    try:
        validate(instance=value, schema=schema)
        logger.debug(f"Results validation passed for job_type: {job_type}")
    except JSONSchemaValidationError as e:
        logger.error(f"Results validation failed for job_type {job_type}: {str(e)}")
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
        ("youtube_frame_extraction", "YouTube Frame Extraction"),
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

    class Meta:
        """Model metadata including database indexes for performance."""
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['created_at']),
            models.Index(fields=['job_type', 'status']),
            models.Index(fields=['user', 'created_at']),
        ]
        ordering = ['-created_at']

    def save(self, *args, **kwargs):
        """Override save to provide instance context for results validation."""
        validate_results_schema.instance = self
        super().save(*args, **kwargs)
        validate_results_schema.instance = None
