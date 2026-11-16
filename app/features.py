"""Turn hourly PM2.5 history + weather into the model's features, exactly as Project 1 built them for training."""

import numpy as np                                       # Maths
import pandas as pd                                      # Tables

WEATHER_RENAME = {"temperature_2m": "temp", "relative_humidity_2m": "humidity", "dew_point_2m": "dew_point",
                  "precipitation": "rain", "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir",
                  "surface_pressure": "pressure", "cloud_cover": "cloud"}   # Open-Meteo names -> P1 training names
WEATHER_VARS = list(WEATHER_RENAME)                      # The variables to request from Open-Meteo


def build_features(pm25: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    """One row per hour with every feature P1 used (except the unavailable gases).
    pm25: hourly station values (local Indian time). weather: hourly Open-Meteo data (local time) that reaches
    at least 24 hours past the last PM2.5 hour, because the model also uses the next day's weather."""
    pm25 = pm25.sort_index().asfreq("h")                 # Missing hours stay empty, exactly like P1's training data
    wx = weather.rename(columns=WEATHER_RENAME).sort_index().asfreq("h")

    d = pd.DataFrame({"pm25": pm25})
    d = d.join(wx[["temp", "humidity", "dew_point", "rain", "wind_speed", "pressure", "cloud"]])   # Weather at the same hour

    d["hour"] = d.index.hour                             # Time features (as in P1)
    d["month"] = d.index.month
    d["dayofweek"] = d.index.day_of_week                # Column keeps P1's name "dayofweek"
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["is_winter"] = d["month"].isin([10, 11, 12, 1]).astype(int)

    rad = np.deg2rad(wx["wind_dir"])                     # Wind as east-west / north-south components
    d["wind_u"] = (wx["wind_speed"] * np.sin(rad)).reindex(d.index)
    d["wind_v"] = (wx["wind_speed"] * np.cos(rad)).reindex(d.index)

    for lag in [1, 2, 3, 6, 12, 24, 48]:                 # PM2.5 history
        d[f"pm25_lag_{lag}h"] = d["pm25"].shift(lag)
    d["pm25_mean_6h"] = d["pm25"].rolling(6, min_periods=4).mean()
    d["pm25_mean_24h"] = d["pm25"].rolling(24, min_periods=18).mean()
    d["pm25_std_24h"] = d["pm25"].rolling(24, min_periods=18).std()
    d["pm25_change_24h"] = d["pm25"] - d["pm25_lag_24h"]

    for col in ["temp", "humidity", "dew_point", "wind_speed", "pressure", "cloud"]:   # The next day's weather
        d[f"{col}_next24"] = wx[col].shift(-24).reindex(d.index)
    d["wind_u_next24"] = (wx["wind_speed"] * np.sin(rad)).shift(-24).reindex(d.index)
    d["wind_v_next24"] = (wx["wind_speed"] * np.cos(rad)).shift(-24).reindex(d.index)
    d["rain_sum_next24"] = wx["rain"].rolling(24).sum().shift(-24).reindex(d.index)
    return d
