from django.shortcuts import render
from django.core.cache import cache

from datetime import datetime, timedelta, timezone as dt_timezone

import requests
import pandas as pd
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY      = os.getenv('API_KEY')
BASE_URL     = 'https://api.openweathermap.org/data/2.5/'
CACHE_TIMEOUT = 300  # Cache API responses for 5 minutes per city

# Module-level model references — populated by ForecastConfig.ready() at startup
rain_model = None
temp_model = None
hum_model  = None
le_dir     = None

# Ordered keyword → safe single-word CSS background class mapping
_DESCRIPTION_CSS_MAP = [
    ('thunderstorm', 'thunder'),
    ('drizzle',      'drizzle'),
    ('rain',         'rain'),
    ('snow',         'snow'),
    ('sleet',        'sleet'),
    ('blizzard',     'blizzard'),
    ('mist',         'mist'),
    ('fog',          'fog'),
    ('smoke',        'smoke'),
    ('haze',         'haze'),
    ('dust',         'haze'),
    ('sand',         'haze'),
    ('ash',          'haze'),
    ('squall',       'thunder'),
    ('tornado',      'thunder'),
    ('overcast',     'overcast'),
    ('clouds',       'clouds'),
    ('clear',        'clear'),
]


def description_to_css_class(description: str) -> str:
    """Map an OWM weather description to a safe single-word CSS class."""
    desc_lower = description.lower()
    for keyword, css_class in _DESCRIPTION_CSS_MAP:
        if keyword in desc_lower:
            return css_class
    return 'clear'


def deg_to_cardinal(deg) -> str:
    """Convert a wind bearing (0–360°) to a 16-point cardinal direction string."""
    try:
        deg = float(deg)
    except (TypeError, ValueError):
        return 'N/A'
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return dirs[round(deg / 22.5) % 16]


def get_current_weather(city: str) -> dict:
    """
    Fetch current weather for a city from the OWM /weather endpoint.
    Returns a data dict on success, or {'error': str} on failure.
    Results are cached for CACHE_TIMEOUT seconds.
    """
    cache_key = f'weather_{city.lower()}'
    cached = cache.get(cache_key)
    if cached:
        logger.debug('Cache hit for city: %s', city)
        return cached

    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.Timeout:
        return {'error': 'Request timed out. Please try again.'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Network error. Please check your internet connection.'}

    if response.status_code == 404:
        return {'error': f"City '{city}' not found. Please check the spelling."}
    if response.status_code == 429:
        return {'error': 'Too many requests to the weather API. Please wait a moment and try again.'}
    if response.status_code != 200:
        return {'error': f"Weather API error (status {response.status_code}). Please try again later."}

    data = response.json()
    try:
        result = {
            'city':            data['name'],
            'country':         data['sys']['country'],
            'current_temp':    round(data['main']['temp']),
            'feels_like':      round(data['main']['feels_like']),
            'temp_min':        round(data['main']['temp_min']),
            'temp_max':        round(data['main']['temp_max']),
            'humidity':        round(data['main']['humidity']),
            'pressure':        data['main']['pressure'],
            'description':     data['weather'][0]['description'],
            'wind_deg':        data['wind'].get('deg', 0),
            'wind_speed':      data['wind'].get('speed', 0),
            'clouds':          data['clouds']['all'],
            'visibility':      data.get('visibility', None),
            'timezone_offset': data.get('timezone', 0),  # UTC offset in seconds
        }
    except KeyError as e:
        return {'error': f"Unexpected API response format (missing field: {e})."}

    cache.set(cache_key, result, CACHE_TIMEOUT)
    return result


def get_owm_forecast(city: str, timezone_offset: int = 0) -> list:
    """
    Fetch the next 5 forecast slots (3-hour intervals) from OWM /forecast endpoint.
    Timestamps are converted to the city's local time using timezone_offset (seconds).
    Returns a list of dicts: [{'time': 'HH:MM', 'temp': float, 'hum': int}, ...]
    """
    cache_key = f'forecast_{city.lower()}'
    cached = cache.get(cache_key)
    if cached:
        return cached

    url = f"{BASE_URL}forecast?q={city}&appid={API_KEY}&units=metric&cnt=5"
    try:
        response = requests.get(url, timeout=10)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return []

    if response.status_code != 200:
        return []

    slots = []
    for item in response.json().get('list', [])[:5]:
        utc_dt   = datetime.fromtimestamp(item['dt'], tz=dt_timezone.utc)
        local_dt = utc_dt + timedelta(seconds=timezone_offset)
        slots.append({
            'time': local_dt.strftime('%H:%M'),
            'temp': round(item['main']['temp'], 1),
            'hum':  round(item['main']['humidity']),
        })

    cache.set(cache_key, slots, CACHE_TIMEOUT)
    return slots


# ── Main view ─────────────────────────────────────────────────────────────────

def weather_view(request):
    """
    Handles POST requests to fetch weather data for a city.
    - Fetches current conditions from OWM /weather
    - Fetches real 3-hourly forecast from OWM /forecast
    - Predicts rain probability using the trained RandomForest classifier
    """
    if request.method != 'POST':
        return render(request, 'weather.html')

    city = request.POST.get('city', '').strip()
    if not city:
        return render(request, 'weather.html', {'error_message': 'Please enter a city name.'})

    current_weather = get_current_weather(city)
    if 'error' in current_weather:
        return render(request, 'weather.html', {'error_message': current_weather['error']})

    # Guard: ensure rain model was loaded at startup
    if rain_model is None:
        return render(request, 'weather.html', {
            'error_message': (
                'Rain prediction model not loaded. '
                'Run train_models.py then restart the server.'
            )
        })

    # ── Rain probability ───────────────────────────────────────────────────────
    rain_features = pd.DataFrame([{
        'Humidity':      current_weather['humidity'],
        'Pressure':      current_weather['pressure'],
        'WindGustSpeed': current_weather['wind_speed'],
        'MinTemp':       current_weather['temp_min'],
    }])
    rain_proba      = rain_model.predict_proba(rain_features)[0]
    rain_yes_index  = list(rain_model.classes_).index(1)
    rain_probability = round(rain_proba[rain_yes_index] * 100)

    # ── Real OWM 5-slot forecast ───────────────────────────────────────────────
    timezone_offset  = current_weather.get('timezone_offset', 0)
    hourly_forecasts = get_owm_forecast(city, timezone_offset)

    # ── Derived display values ─────────────────────────────────────────────────
    wind_direction = deg_to_cardinal(current_weather['wind_deg'])
    css_class      = description_to_css_class(current_weather['description'])
    visibility_str = (
        f"{current_weather['visibility']} m"
        if current_weather['visibility'] is not None else 'N/A'
    )

    logger.debug(
        'Rain probability: %d%% | Forecast slots: %d | CSS class: %s',
        rain_probability, len(hourly_forecasts), css_class
    )

    context = {
        'location':    city,
        'css_class':   css_class,

        'current_temp': f"{current_weather['current_temp']}°C",
        'MinTemp':      f"{current_weather['temp_min']}°C",
        'MaxTemp':      f"{current_weather['temp_max']}°C",
        'feels_like':   f"{current_weather['feels_like']}°C",
        'humidity':     f"{current_weather['humidity']}%",
        'clouds':       f"{current_weather['clouds']}%",
        'description':  current_weather['description'],

        'city':    current_weather['city'],
        'country': current_weather['country'],
        'time':    datetime.now(dt_timezone.utc),
        'date':    datetime.now(dt_timezone.utc).strftime('%B %d, %Y'),

        'wind':       f"{current_weather['wind_speed']} m/s {wind_direction}",
        'pressure':   f"{current_weather['pressure']} hPa",
        'visibility': visibility_str,

        'hourly_forecasts': hourly_forecasts,
        'rain_probability': rain_probability,
    }

    return render(request, 'weather.html', context)
