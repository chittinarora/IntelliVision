from rest_framework.routers import DefaultRouter
from .views import VideoJobViewSet
from django.urls import path, include

router = DefaultRouter()
router.register(r'jobs', VideoJobViewSet, basename='videojob')

urlpatterns = router.urls + [
    path('api/faceauth/', include('faceauth.urls')),
]
