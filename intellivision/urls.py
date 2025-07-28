"""
URL configuration for intellivision project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse
from rest_framework_simplejwt.views import TokenRefreshView


def root(request):
    # Return a simple welcome message as an HTTP response.
    # The content must be bytes, not str, to avoid type errors.
    return HttpResponse(b"Welcome to the People Tracker API! Go to /api/jobs/ or /admin/")

def health_check(request):
    """Health check endpoint for Docker health checks."""
    return HttpResponse(b"OK", content_type="text/plain")

urlpatterns = [
    path('', root),
    path('health/', health_check),
    path('admin/', admin.site.urls),
    path('api/', include('apps.video_analytics.urls')),
    path('api/faceauth/', include('apps.face_auth.urls')),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
