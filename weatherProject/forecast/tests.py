"""
Unit and integration tests for the forecast app.

Run from the project root:
    python manage.py test forecast
"""

from django.test import TestCase, Client
from unittest.mock import patch, MagicMock

from forecast import views


# ── Helper: mock OWM /weather response payload ────────────────────────────────

def _mock_weather_payload():
    return {
        'name': 'London',
        'sys':  {'country': 'GB'},
        'main': {
            'temp': 15.5, 'feels_like': 13.0,
            'temp_min': 12.0, 'temp_max': 18.5,
            'humidity': 80, 'pressure': 1015,
        },
        'weather':    [{'description': 'clear sky'}],
        'wind':       {'deg': 180, 'speed': 5.0},
        'clouds':     {'all': 20},
        'visibility': 10000,
        'timezone':   3600,
    }


def _mock_current_weather():
    """Return a pre-built current_weather dict (bypassing the API call)."""
    return {
        'city': 'London', 'country': 'GB',
        'current_temp': 16, 'feels_like': 13, 'temp_min': 12, 'temp_max': 19,
        'humidity': 80, 'pressure': 1015, 'description': 'clear sky',
        'wind_deg': 180, 'wind_speed': 5.0,
        'clouds': 20, 'visibility': 10000, 'timezone_offset': 3600,
    }


def _mock_forecast():
    return [
        {'time': '14:00', 'temp': 15.0, 'hum': 78},
        {'time': '17:00', 'temp': 14.5, 'hum': 82},
        {'time': '20:00', 'temp': 13.0, 'hum': 85},
        {'time': '23:00', 'temp': 12.0, 'hum': 87},
        {'time': '02:00', 'temp': 11.5, 'hum': 90},
    ]


# ── 1. get_current_weather() unit tests ───────────────────────────────────────

class GetCurrentWeatherTests(TestCase):

    def setUp(self):
        # Clear the cache before every test so cached API results
        # from previous tests don't bleed through.
        from django.core.cache import cache
        cache.clear()

    @patch('forecast.views.requests.get')
    def test_success_returns_expected_keys(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _mock_weather_payload()
        mock_get.return_value = mock_resp

        result = views.get_current_weather('London')

        self.assertNotIn('error', result)
        self.assertEqual(result['city'], 'London')
        self.assertEqual(result['country'], 'GB')
        self.assertEqual(result['current_temp'], 16)   # round(15.5)
        self.assertEqual(result['timezone_offset'], 3600)

    @patch('forecast.views.requests.get')
    def test_city_not_found_returns_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = views.get_current_weather('NonExistentCity999')

        self.assertIn('error', result)
        self.assertIn('not found', result['error'])

    @patch('forecast.views.requests.get')
    def test_timeout_returns_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout()

        result = views.get_current_weather('Tokyo')   # distinct key — no cache collision

        self.assertIn('error', result)
        self.assertIn('timed out', result['error'])

    @patch('forecast.views.requests.get')
    def test_connection_error_returns_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError()

        result = views.get_current_weather('Paris')   # distinct key — no cache collision

        self.assertIn('error', result)
        self.assertIn('Network error', result['error'])

    @patch('forecast.views.requests.get')
    def test_rate_limit_returns_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp

        result = views.get_current_weather('Berlin')  # distinct key — no cache collision

        self.assertIn('error', result)
        self.assertIn('Too many requests', result['error'])


# ── 2. Helper function unit tests ─────────────────────────────────────────────

class HelperFunctionTests(TestCase):

    # deg_to_cardinal
    def test_cardinal_north(self):
        self.assertEqual(views.deg_to_cardinal(0),   'N')
        self.assertEqual(views.deg_to_cardinal(360), 'N')

    def test_cardinal_east(self):
        self.assertEqual(views.deg_to_cardinal(90), 'E')

    def test_cardinal_south(self):
        self.assertEqual(views.deg_to_cardinal(180), 'S')

    def test_cardinal_west(self):
        self.assertEqual(views.deg_to_cardinal(270), 'W')

    def test_cardinal_invalid_returns_na(self):
        self.assertEqual(views.deg_to_cardinal('N/A'), 'N/A')
        self.assertEqual(views.deg_to_cardinal(None),  'N/A')

    # description_to_css_class
    def test_css_class_light_rain(self):
        self.assertEqual(views.description_to_css_class('light rain'), 'rain')

    def test_css_class_clear_sky(self):
        self.assertEqual(views.description_to_css_class('clear sky'), 'clear')

    def test_css_class_broken_clouds(self):
        self.assertEqual(views.description_to_css_class('broken clouds'), 'clouds')

    def test_css_class_thunderstorm(self):
        self.assertEqual(views.description_to_css_class('thunderstorm with rain'), 'thunder')

    def test_css_class_unknown_defaults_to_clear(self):
        self.assertEqual(views.description_to_css_class('unknown condition xyz'), 'clear')


# ── 3. weather_view integration tests ────────────────────────────────────────

class WeatherViewTests(TestCase):

    def setUp(self):
        self.client = Client()

    def test_get_returns_200(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_post_empty_city_shows_error(self):
        response = self.client.post('/', {'city': ''})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Please enter a city name')

    def test_post_whitespace_only_city_shows_error(self):
        response = self.client.post('/', {'city': '   '})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Please enter a city name')

    @patch('forecast.views.get_current_weather')
    def test_post_api_error_shows_error(self, mock_weather):
        mock_weather.return_value = {'error': 'City not found.'}

        response = self.client.post('/', {'city': 'FakeCity'})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'City not found.')

    @patch('forecast.views.get_owm_forecast')
    @patch('forecast.views.get_current_weather')
    def test_post_valid_city_renders_context(self, mock_weather, mock_forecast):
        mock_weather.return_value  = _mock_current_weather()
        mock_forecast.return_value = _mock_forecast()

        mock_rain = MagicMock()
        mock_rain.predict_proba.return_value = [[0.35, 0.65]]
        mock_rain.classes_ = [0, 1]
        views.rain_model = mock_rain

        response = self.client.post('/', {'city': 'London'})

        self.assertEqual(response.status_code, 200)
        self.assertIn('rain_probability', response.context)
        self.assertEqual(response.context['rain_probability'], 65)
        self.assertEqual(response.context['city'], 'London')
        self.assertEqual(response.context['css_class'], 'clear')
        self.assertEqual(len(response.context['hourly_forecasts']), 5)

    @patch('forecast.views.get_owm_forecast')
    @patch('forecast.views.get_current_weather')
    def test_post_no_model_shows_error(self, mock_weather, mock_forecast):
        mock_weather.return_value  = _mock_current_weather()
        mock_forecast.return_value = _mock_forecast()
        views.rain_model = None   # Simulate model not loaded

        response = self.client.post('/', {'city': 'London'})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'not loaded')
