"""
urls.py - Face Auth App
URL patterns for face authentication endpoints.
"""

from django.urls import path
from .views import RegisterFaceView, LoginFaceView, TaskStatusView

urlpatterns = [
    path('register-face/', RegisterFaceView.as_view(), name='register-face'),
    path('login-face/', LoginFaceView.as_view(), name='login-face'),
    path('task-status/<str:task_id>/', TaskStatusView.as_view(), name='task-status'),
]
