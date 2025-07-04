from rest_framework.routers import DefaultRouter
from .views import VideoJobViewSet, current_user_view, food_waste_estimation_view, pothole_detection_image_view, pothole_detection_video_view, pest_monitoring_image_view, pest_monitoring_video_view, car_count_view, anpr_callback_view, emergency_count_view, room_readiness_view, lobby_detection_view
from django.urls import path, include

"""
URL configuration for the tracker app. Registers routes for jobs and user info endpoints.
"""

router = DefaultRouter()
router.register(r'jobs', VideoJobViewSet, basename='videojob')

# Combine router URLs with custom endpoints
urlpatterns = router.urls + [
    path('me/', current_user_view),  # Endpoint for current user info
    path('food-waste-estimation/', food_waste_estimation_view),  # Endpoint for food waste estimation
    path('faceauth/', include('apps.face_auth.urls')),
    path('pothole-detection/video/', pothole_detection_video_view, name='pothole_detection_video'),
    path('pothole-detection/image/', pothole_detection_image_view, name='pothole_detection_image'),
    path('pest-monitoring/image/', pest_monitoring_image_view, name='pest_monitoring_image'),
    path('pest-monitoring/video/', pest_monitoring_video_view, name='pest_monitoring_video'),
    path('car-count/', car_count_view, name='car_count'),
    path('api/anpr-callback/', anpr_callback_view, name='anpr_callback'),
    path('emergency-count/', emergency_count_view, name='emergency_count'),
    path('room-readiness/', room_readiness_view, name='room_readiness'),
    path('lobby-detection/', lobby_detection_view, name='lobby-detection'),
]
