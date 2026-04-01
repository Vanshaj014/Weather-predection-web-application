import os
import logging
import joblib
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ForecastConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'forecast'

    def ready(self):
        """
        Load ML models once at startup to avoid per-request disk I/O.
        Failures are logged as errors rather than crashing the server,
        so the model-not-loaded guard in views.py can handle them gracefully.
        """
        from . import views

        BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.join(BASE_DIR, 'models')

        try:
            views.rain_model = joblib.load(os.path.join(MODELS_DIR, 'rain_model.joblib'))
            views.temp_model = joblib.load(os.path.join(MODELS_DIR, 'temp_model.joblib'))
            views.hum_model  = joblib.load(os.path.join(MODELS_DIR, 'hum_model.joblib'))
            views.le_dir     = joblib.load(os.path.join(MODELS_DIR, 'le_dir.joblib'))
            logger.info('All ML models loaded successfully from %s', MODELS_DIR)
        except FileNotFoundError as exc:
            logger.error(
                'Model file not found: %s — Run train_models.py to generate models.',
                exc
            )
        except Exception as exc:  # noqa: BLE001
            logger.error('Unexpected error while loading ML models: %s', exc)
