from rest_framework import serializers
from .models import VideoJob


class VideoJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoJob
        fields = '__all__'
        read_only_fields = ('user', 'status', 'output_video', 'created_at', 'updated_at')
