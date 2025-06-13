from rest_framework.routers import DefaultRouter
from .views import VideoJobViewSet

router = DefaultRouter()
router.register(r'jobs', VideoJobViewSet, basename='videojob')

urlpatterns = router.urls
