# /apps/video_analytics/urls.py

from rest_framework.routers import DefaultRouter
from django.urls import path, include
from .views import (
    VideoJobViewSet, current_user_view, food_waste_estimation_view,
    pothole_detection_image_view, wildlife_detection_image_view,
    wildlife_detection_video_view, car_count_view, parking_analysis_view,
    emergency_count_view, room_readiness_view, lobby_detection_view,
    get_youtube_frame_view, people_count_view
)

"""
URL configuration for the video analytics app. Defines routes for job management,
user info, and specific analytics endpoints.
"""

router = DefaultRouter()
router.register(r'jobs', VideoJobViewSet, basename='videojob')

urlpatterns = router.urls + [
    path('car-count/', car_count_view, name='car_count'),
    path('emergency-count/', emergency_count_view, name='emergency_count'),
    path('faceauth/', include('apps.face_auth.urls')),
    path('food-waste-estimation/', food_waste_estimation_view, name='food_waste_estimation'),
    path('get-youtube-frame/', get_youtube_frame_view, name='get_youtube_frame'),
    path('lobby-detection/', lobby_detection_view, name='lobby_detection'),
    path('parking-analysis/', parking_analysis_view, name='parking_analysis'),
    path('people-count/', people_count_view, name='people_count'),
    path('pothole-detection/image/', pothole_detection_image_view, name='pothole_detection_image'),
    path('room-readiness/', room_readiness_view, name='room_readiness'),
    path('wildlife-detection/image/', wildlife_detection_image_view, name='wildlife_detection_image'),
    path('wildlife-detection/video/', wildlife_detection_video_view, name='wildlife_detection_video'),
    path('me/', current_user_view, name='current_user'),
]