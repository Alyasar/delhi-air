"""Train/serve parity: the live feature code must reproduce Project 1's training features exactly."""

from pathlib import Path                                 # File paths
import numpy as np                                       # Maths
import pandas as pd                                      # Tables

from app.features import build_features                  # The code the live API uses

DATA = Path(__file__).resolve().parent.parent / "data"


def test_live_features_match_training_features():
    merged = pd.read_csv(DATA / "merged.csv", index_col="time", parse_dates=True)     # P1's hourly data (before features)
    training = pd.read_csv(DATA / "features.csv", index_col="time", parse_dates=True)  # P1's training features
    weather = pd.read_csv(DATA / "weather.csv", skiprows=3, index_col="time", parse_dates=True).iloc[:, :8]
    weather.columns = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
                       "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover"]   # Open-Meteo API names

    live = build_features(merged["pm25"], weather)       # Exactly what the API computes, on the training history
    hours = training.index.intersection(live.index)
    shared = [c for c in training.columns if c in live.columns]
    assert len(hours) == len(training) and len(shared) >= 29
    for col in shared:                                   # Every shared feature, every hour, must be identical
        diff = np.nanmax(np.abs(training.loc[hours, col].to_numpy() - live.loc[hours, col].to_numpy()))
        assert diff < 1e-9, f"{col} differs by {diff}"
