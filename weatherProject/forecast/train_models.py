import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'weather.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)


def read_historical_data(filename):
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def prepare_data(data):
    le_dir  = LabelEncoder()
    le_rain = LabelEncoder()

    data['WindGustDir']   = le_dir.fit_transform(data['WindGustDir'])
    data['RainTomorrow']  = le_rain.fit_transform(data['RainTomorrow'])

    X = data[['Humidity', 'Pressure', 'WindGustSpeed', 'MinTemp']]
    y = data['RainTomorrow']

    return X, y, le_dir, le_rain


def train_rain_model(x, y):
    param_grid = {
        'n_estimators':      [50, 100, 200],
        'max_depth':         [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x, y)

    print("Best parameters for rain model:", grid_search.best_params_)
    return grid_search.best_estimator_


def prepare_regression_data(data, feature, window_size=3):
    x, y = [], []
    for i in range(len(data) - window_size):
        window = data[feature].iloc[i:i + window_size].values
        target = data[feature].iloc[i + window_size]
        x.append(window)
        y.append(target)

    return np.array(x), np.array(y)


def train_regression_model(x, y, label=''):
    # Proper 80/20 train/test split so MAE reflects generalisation, not overfitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    train_mae = mean_absolute_error(y_train, model.predict(x_train))
    test_mae  = mean_absolute_error(y_test,  model.predict(x_test))
    print(f"  {label} — Train MAE: {train_mae:.2f}  |  Test MAE: {test_mae:.2f}")

    return model


def main():
    print("Loading historical data...")
    historical_data = read_historical_data(DATA_PATH)

    # --- Rain Prediction Model ---
    print("\nTraining rain prediction model with GridSearchCV...")
    rain_data = historical_data.copy()
    x_rain, y_rain, le_dir, le_rain = prepare_data(rain_data)
    rain_model = train_rain_model(x_rain, y_rain)
    joblib.dump(rain_model, os.path.join(MODELS_DIR, 'rain_model.joblib'))
    joblib.dump(le_dir,     os.path.join(MODELS_DIR, 'le_dir.joblib'))
    joblib.dump(le_rain,    os.path.join(MODELS_DIR, 'le_rain.joblib'))
    print("Rain prediction model saved.")

    # --- Temperature Regression Model ---
    print("\nTraining temperature regression model...")
    x_temp, y_temp = prepare_regression_data(historical_data, 'Temp', window_size=3)
    temp_model = train_regression_model(x_temp, y_temp, label='Temperature')
    joblib.dump(temp_model, os.path.join(MODELS_DIR, 'temp_model.joblib'))
    print("Temperature regression model saved.")

    # --- Humidity Regression Model ---
    print("\nTraining humidity regression model...")
    x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity', window_size=3)
    hum_model = train_regression_model(x_hum, y_hum, label='Humidity')
    joblib.dump(hum_model, os.path.join(MODELS_DIR, 'hum_model.joblib'))
    print("Humidity regression model saved.")

    print("\nAll models trained and saved successfully.")


if __name__ == '__main__':
    main()