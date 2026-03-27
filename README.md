# Weather Prediction Django Project

A Django-based web application that provides current weather information and ML-powered forecasts for any city. It uses the OpenWeatherMap API for real-time data and scikit-learn to predict rain tomorrow and forecast temperature/humidity over the next 5 hours.

## Features

- **Current Weather:** Temperature, feels-like, humidity, cloud cover, wind speed, pressure, and visibility.
- **Rain Prediction:** Predicts whether it will rain tomorrow using a tuned Random Forest Classifier.
- **5-Hour Forecast:** Temperature and humidity forecast using autoregressive Random Forest Regressors.
- **Interactive Chart:** Chart.js visualization of the 5-hour forecast.
- **Dynamic Background:** Background changes based on current weather conditions.

## Project Structure

```
Weather prediction/
├── evaluate_models.py      # Standalone script to evaluate model accuracy
├── requirements.txt
├── .gitignore
├── .env                    # NOT in repo — create manually (see Setup)
└── weatherProject/
    ├── manage.py
    ├── forecast/           # Main Django app
    │   ├── data/
    │   │   └── weather.csv         # Historical training data
    │   ├── models/                 # Saved ML models (NOT in repo — retrain locally)
    │   ├── static/                 # CSS, JS, images
    │   ├── templates/
    │   │   └── weather.html
    │   ├── apps.py                 # Loads ML models once at startup
    │   ├── train_models.py         # Offline training script
    │   └── views.py
    └── weatherProject/             # Django settings
```

## ML Models

| Model | Algorithm | Key Metric |
|---|---|---|
| Rain prediction | RandomForestClassifier + GridSearchCV + `class_weight='balanced'` | 93%+ accuracy |
| Temperature forecast | RandomForestRegressor (autoregressive, window=3) | ~1°C MAE |
| Humidity forecast | RandomForestRegressor (autoregressive, window=3) | ~4.6% MAE |

## Setup and Installation

### Prerequisites

- Python 3.10+
- An API key from [OpenWeatherMap](https://openweathermap.org/api)

### 1. Clone the repository

```bash
git clone <repository-url>
cd "Weather prediction"
```

### 2. Create and activate a virtual environment

```powershell
# Windows
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the `.env` file

Create a file named `.env` in the project root with the following content:

```env
API_KEY="YOUR_OPENWEATHERMAP_API_KEY"
SECRET_KEY="YOUR_DJANGO_SECRET_KEY"
DEBUG=True
ALLOWED_HOSTS=*
```

Generate a Django secret key with:
```python
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 5. Train the ML models

The trained model files are not included in the repository. Run this once before starting the server:

```bash
python weatherProject/forecast/train_models.py
```

This saves the models to `weatherProject/forecast/models/`.

### 6. Run the development server

```powershell
cd weatherProject
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

## Evaluating Model Accuracy

```bash
python evaluate_models.py
```

This runs all three models against a held-out 20% test set and reports accuracy, precision, recall, F1 (rain model) and MAE, R² (regression models).
