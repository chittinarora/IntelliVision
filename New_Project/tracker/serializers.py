from rest_framework import serializers
from .models import VideoJob


"""
Serializers for the tracker app. Handles serialization of VideoJob model instances.
"""


class VideoJobSerializer(serializers.ModelSerializer):
    """
    Serializer for the VideoJob model. Handles validation and serialization of job fields.
    """
    class Meta:
        model = VideoJob
        fields = '__all__'
        read_only_fields = ('user', 'status', 'output_video', 'created_at', 'updated_at')


class FoodWasteEstimationSerializer(serializers.Serializer):
    image = serializers.ImageField()
