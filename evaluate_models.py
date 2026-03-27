"""
Model Accuracy Evaluation Script
Run from project root: python evaluate_models.py
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, mean_absolute_error, r2_score
)

BASE_DIR   = os.path.join('weatherProject', 'forecast')
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'weather.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(f"Dataset: {len(df)} rows\n")
print("=" * 55)

# ── 1. Rain Prediction (Classification) ────────────────────────────────────
print("\n🌧  RAIN PREDICTION MODEL (RandomForestClassifier)")
print("-" * 55)

le_dir  = LabelEncoder()
le_rain = LabelEncoder()

rain_df = df.copy()
rain_df['WindGustDir']  = le_dir.fit_transform(rain_df['WindGustDir'])
rain_df['RainTomorrow'] = le_rain.fit_transform(rain_df['RainTomorrow'])

X = rain_df[['Humidity', 'Pressure', 'WindGustSpeed', 'MinTemp']]
y = rain_df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rain_model = joblib.load(os.path.join(MODELS_DIR, 'rain_model.joblib'))
y_pred     = rain_model.predict(X_test)

print(f"  Test samples : {len(y_test)}")
print(f"  Accuracy     : {accuracy_score(y_test, y_pred):.4f}  ({accuracy_score(y_test, y_pred)*100:.1f}%)")
print(f"  Precision    : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Recall       : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1 Score     : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print()
print(classification_report(y_test, y_pred,
                            target_names=le_rain.classes_, zero_division=0))

# ── Helper: sliding-window regression data ──────────────────────────────────
def prepare_regression_data(data, feature, window_size=3):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[feature].iloc[i:i + window_size].values)
        y.append(data[feature].iloc[i + window_size])
    return np.array(x), np.array(y)

# ── 2. Temperature Regression ───────────────────────────────────────────────
print("=" * 55)
print("\n🌡  TEMPERATURE MODEL (RandomForestRegressor)")
print("-" * 55)

x_temp, y_temp = prepare_regression_data(df, 'Temp', window_size=3)
x_tr, x_te, y_tr, y_te = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)

temp_model = joblib.load(os.path.join(MODELS_DIR, 'temp_model.joblib'))
y_pred_temp = temp_model.predict(x_te)

print(f"  Test samples : {len(y_te)}")
print(f"  MAE          : {mean_absolute_error(y_te, y_pred_temp):.4f} °C")
print(f"  R² Score     : {r2_score(y_te, y_pred_temp):.4f}")

# ── 3. Humidity Regression ──────────────────────────────────────────────────
print()
print("=" * 55)
print("\n💧  HUMIDITY MODEL (RandomForestRegressor)")
print("-" * 55)

x_hum, y_hum = prepare_regression_data(df, 'Humidity', window_size=3)
x_tr, x_te, y_tr, y_te = train_test_split(x_hum, y_hum, test_size=0.2, random_state=42)

hum_model  = joblib.load(os.path.join(MODELS_DIR, 'hum_model.joblib'))
y_pred_hum = hum_model.predict(x_te)

print(f"  Test samples : {len(y_te)}")
print(f"  MAE          : {mean_absolute_error(y_te, y_pred_hum):.4f} %")
print(f"  R² Score     : {r2_score(y_te, y_pred_hum):.4f}")

print("\n" + "=" * 55)
print("Evaluation complete.")
