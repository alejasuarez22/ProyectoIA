import joblib
import numpy as np
import pandas as pd

def load_model(path):
  return joblib.load(path)

def prepare(df, art):
  df = df.copy()
  df['datetime'] = pd.to_datetime(df['datetime'])
  df['weather'] = df['weather'].replace(4, 3)
  df['windspeed'] = df['windspeed'].replace(0, np.nan)
  df['humidity'] = df['humidity'].replace(0, np.nan)
  df['hour'] = df['datetime'].dt.hour
  df['dayofweek'] = df['datetime'].dt.dayofweek
  df['month'] = df['datetime'].dt.month
  df['year'] = df['datetime'].dt.year
  df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
  df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
  df[['season', 'weather']] = df[['season', 'weather']].astype(str)
  return df[art['features']]

def predict(df, art):
  X = prepare(df, art)
  return np.clip(np.expm1(art['model'].predict(X)), 0, None)
