from rest_framework.routers import DefaultRouter
from .views import VideoJobViewSet, current_user_view, food_waste_estimation_view, pothole_detection_image_view, wildlife_detection_image_view, wildlife_detection_video_view, car_count_view, parking_analysis_view, emergency_count_view, room_readiness_view, lobby_detection_view, AnalyzeYouTubeView
from django.urls import path, include

"""
URL configuration for the tracker app. Registers routes for jobs and user info endpoints.
"""

router = DefaultRouter()
router.register(r'jobs', VideoJobViewSet, basename='videojob')

# Combine router URLs with custom endpoints
urlpatterns = router.urls + [
    path('me/', current_user_view),  # Endpoint for current user info
    path('food-waste-estimation/', food_waste_estimation_view),
    path('faceauth/', include('apps.face_auth.urls')),
    path('pothole-detection/image/', pothole_detection_image_view, name='pothole_detection_image'),
    path('wildlife-detection/image/', wildlife_detection_image_view, name='wildlife_detection_image'),
    path('wildlife-detection/video/', wildlife_detection_video_view, name='wildlife_detection_video'),
    path('car-count/', car_count_view, name='car_count_view'),
    path('parking-analysis/', parking_analysis_view, name='parking_analysis_view'),
    path('emergency-count/', emergency_count_view, name='emergency_count'),
    path('room-readiness/', room_readiness_view, name='room_readiness'),
    path('lobby-detection/', lobby_detection_view, name='lobby-detection'),
    path('analyze-youtube/', AnalyzeYouTubeView.as_view(), name='analyze-youtube'),
]
