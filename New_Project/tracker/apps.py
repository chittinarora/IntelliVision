from django.apps import AppConfig


class TrackerConfig(AppConfig):
    """
    Configuration for the tracker app.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tracker'
