# /apps/video_analytics/serializers.py

"""
=====================================
Imports
=====================================
Serializers for the video analytics app, handling VideoJob and related data.
Added results validation using jsonschema (Issue #7).
"""

from rest_framework import serializers
from .models import VideoJob, RESULT_SCHEMAS
from jsonschema import validate, ValidationError as JSONSchemaValidationError

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

"""
=====================================
VideoJob Serializer
=====================================
Serializer for VideoJob model, with validation for file size, type, and results.
Added validate_results to enforce job-type-specific schemas (Issue #7).
"""

class VideoJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoJob
        fields = '__all__'
        read_only_fields = ('user', 'status', 'output_video', 'output_image', 'created_at', 'updated_at')

    def validate_input_video(self, value):
        """Validate input file size and type."""
        if value.size > MAX_FILE_SIZE:
            raise serializers.ValidationError(f"File size {value.size / (1024*1024):.2f}MB exceeds 500MB limit")
        return value

    def validate_results(self, value):
        """Validate results JSON against job-type-specific schema."""
        if value is None:
            return value
        job_type = self.initial_data.get('job_type', self.instance.job_type if self.instance else None)
        if not job_type:
            raise serializers.ValidationError("job_type is required for results validation")
        schema = RESULT_SCHEMAS.get(job_type)
        if not schema:
            raise serializers.ValidationError(f"No schema defined for job_type: {job_type}")
        try:
            validate(instance=value, schema=schema)
        except JSONSchemaValidationError as e:
            raise serializers.ValidationError(f"Invalid results format for {job_type}: {str(e)}")
        return value

"""
=====================================
FoodWasteEstimation Serializer
=====================================
Serializer for food waste estimation image upload.
No changes needed for Issue #7.
"""

class FoodWasteEstimationSerializer(serializers.Serializer):
    image = serializers.ImageField()

    def validate_image(self, value):
        """Validate image file size and type."""
        if value.size > MAX_FILE_SIZE:
            raise serializers.ValidationError(f"File size {value.size / (1024*1024):.2f}MB exceeds 500MB limit")
        return value