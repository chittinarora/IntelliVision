from rest_framework.routers import DefaultRouter
from .views import VideoJobViewSet, current_user_view
from django.urls import path, include

router = DefaultRouter()
router.register(r'jobs', VideoJobViewSet, basename='videojob')

urlpatterns = router.urls + [
    path('me/', current_user_view),
    path('faceauth/', include('faceauth.urls')),
]
