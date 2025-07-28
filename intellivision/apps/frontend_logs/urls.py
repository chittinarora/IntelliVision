"""
URL patterns for frontend logging endpoints.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.receive_frontend_logs, name='frontend_logs'),
    path('health/', views.health_check, name='frontend_logs_health'),
]