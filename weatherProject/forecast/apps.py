import os
import joblib
from django.apps import AppConfig


class ForecastConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'forecast'

    def ready(self):
        """Load ML models once at startup to avoid per-request disk I/O."""
        from . import views

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.join(BASE_DIR, 'models')

        views.rain_model = joblib.load(os.path.join(MODELS_DIR, 'rain_model.joblib'))
        views.temp_model = joblib.load(os.path.join(MODELS_DIR, 'temp_model.joblib'))
        views.hum_model  = joblib.load(os.path.join(MODELS_DIR, 'hum_model.joblib'))
        views.le_dir     = joblib.load(os.path.join(MODELS_DIR, 'le_dir.joblib'))
