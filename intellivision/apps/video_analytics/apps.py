# /apps/video_analytics/apps.py

from django.apps import AppConfig
import logging

"""
App configuration for the video analytics Django app.
"""

logger = logging.getLogger(__name__)

class TrackerConfig(AppConfig):
    """Configuration for the video analytics app."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.video_analytics'
    verbose_name = 'Video Analytics'

    def ready(self):
        """Initialize the app when Django starts."""
        # Only run initialization once
        if not hasattr(self, '_initialized'):
            try:
                # Initialize model management system
                from .analytics.model_manager import initialize_models
                logger.info(" Initializing video analytics model management system...")
                model_status = initialize_models(auto_download=True)
                logger.info(f"SUCCESS: Model management initialized: {sum(model_status.values())}/{len(model_status)} models available")
                self._initialized = True
            except Exception as e:
                logger.error(f"ERROR: Failed to initialize model management system: {e}")
                # Don't fail app startup, just log the error
