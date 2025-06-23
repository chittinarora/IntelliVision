from rest_framework.routers import DefaultRouter
from .views import VideoJobViewSet, current_user_view, food_waste_estimation_view
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
    path('faceauth/', include('faceauth.urls')),
]
