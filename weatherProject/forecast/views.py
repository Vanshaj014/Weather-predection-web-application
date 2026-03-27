from django.shortcuts import render

from datetime import datetime, timedelta
import pytz

import requests
import pandas as pd
import numpy as np
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv('API_KEY')
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Module-level model references — populated by ForecastConfig.ready() at startup
rain_model = None
temp_model = None
hum_model  = None
le_dir     = None


# Fetch CURRENT Weather Data
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        return {'error': f"Failed to retrieve data. Status code: {response.status_code}"}

    data = response.json()

    try:
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'wind_gust_dir': data['wind'].get('deg', 'N/A'),
            'wind_gust_speed': data['wind'].get('speed', 'N/A'),
            'clouds': data['clouds']['all'],
            'visibility': data['visibility'],
        }
    except KeyError as e:
        return {'error': f"Missing expected data in response: {e}"}


# Read Historical Data
def read_historical_data(filename):
    """Reads a CSV file, removes missing and duplicate entries, and returns a cleaned DataFrame."""
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


# Predict future values via sliding window
def predict_future(model, history, steps=5, round_result=False):
    """
    Predicts future values recursively using a sliding window model.

    Parameters:
    - model: Trained regression model that expects a window of previous values.
    - history: A list of recent values to start the prediction from.
    - steps: Number of future steps to predict.
    - round_result: Whether to round predictions.

    Returns:
    - List of predicted values.
    """
    predictions = []
    current_window = list(history)

    for _ in range(steps):
        window_array = np.array(current_window).reshape(1, -1)
        next_value = model.predict(window_array)[0]
        if round_result:
            next_value = round(next_value, 2)
        predictions.append(next_value)
        current_window.pop(0)
        current_window.append(next_value)

    return predictions


# Weather analysis view
def weather_view(request):
    """
    Handles user POST request to analyze weather for a city.
    Fetches current weather, predicts rain, and forecasts future temp & humidity.
    Renders the result in a template.
    """

    if request.method == 'POST':
        city = request.POST.get('city', '').strip()

        # Input validation
        if not city:
            context = {'error_message': 'Please enter a city name.'}
            return render(request, 'weather.html', context)

        current_weather = get_current_weather(city)

        # Handle API errors gracefully
        if 'error' in current_weather:
            context = {'error_message': current_weather['error']}
            return render(request, 'weather.html', context)

        # Guard: ensure models were loaded at startup
        if rain_model is None or temp_model is None or hum_model is None:
            context = {'error_message': 'Models not loaded. Please run train_models.py first, then restart the server.'}
            return render(request, 'weather.html', context)

        # --- Rain Prediction ---
        rain_features = pd.DataFrame([{
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'WindGustSpeed': current_weather['wind_gust_speed'],
            'MinTemp': current_weather['temp_min'],
        }])
        rain_prediction = rain_model.predict(rain_features)[0]

        # --- Temp & Humidity Forecast ---
        window_size = 3  # Must match the window_size used in training

        temp_history = [current_weather['current_temp']] * window_size
        hum_history  = [current_weather['humidity']] * window_size

        future_temp     = predict_future(temp_model, temp_history, steps=5, round_result=True)
        future_humidity = predict_future(hum_model,  hum_history,  steps=5, round_result=True)

        # Generate future time labels in UTC
        now       = datetime.now(pytz.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        feature_times = [(next_hour + timedelta(hours=i)).strftime("%H:00 UTC") for i in range(5)]

        hourly_forecasts = [
            {'time': feature_times[i], 'temp': future_temp[i], 'hum': future_humidity[i]}
            for i in range(5)
        ]

        logger.debug('Hourly Forecasts: %s', hourly_forecasts)

        context = {
            'location': city,
            'current_temp': f"{current_weather['current_temp']}°C",
            'MinTemp': f"{current_weather['temp_min']}°C",
            'MaxTemp': f"{current_weather['temp_max']}°C",
            'feels_like': f"{current_weather['feels_like']}°C",
            'humidity': f"{current_weather['humidity']}%",
            'clouds': f"{current_weather['clouds']}%",
            'description': current_weather['description'],

            'city': current_weather['city'],
            'country': current_weather['country'],
            'time': datetime.now(pytz.utc),
            'date': datetime.now(pytz.utc).strftime("%B %d, %Y"),

            'wind': f"{current_weather['wind_gust_speed']} m/s",
            'pressure': f"{current_weather['pressure']} hPa",
            'visibility': f"{current_weather['visibility']} m",

            'hourly_forecasts': hourly_forecasts,
            'rain_prediction': 'Yes' if rain_prediction == 1 else 'No',
        }

        return render(request, 'weather.html', context)

    return render(request, 'weather.html')
