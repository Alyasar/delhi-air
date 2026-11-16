"""Live data: PM2.5 measured at Shadipur (OpenAQ), the weather forecast and the CAMS estimate (Open-Meteo)."""

import os                                                # Read the OpenAQ key from the environment
from datetime import datetime, timedelta, timezone       # Time windows
import httpx                                             # HTTP client
import pandas as pd                                      # Tables

from app.features import WEATHER_VARS                    # The weather variables the model was trained on

OPENAQ = "https://api.openaq.org/v3"
LOCATION_ID = 5630                                       # Shadipur, Delhi - CPCB (the live feed; 2503 is a dead 2018 feed)
STATION_POINT = (28.6517, 77.1582)                       # Shadipur station
WEATHER_POINT = (28.646748, 77.17218)                    # The Open-Meteo grid point P1 trained on
TIMEOUT = 20                                             # Seconds per request
VALID_PM25 = (0, 1500)                                   # Values outside this range are sensor errors


def _openaq_headers():
    key = os.environ.get("OPENAQ_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAQ_API_KEY is not set")
    return {"X-API-Key": key}


def _hours_to_series(results):
    """OpenAQ hourly results -> a Series indexed by local (Indian) time."""
    values = {pd.Timestamp(h["period"]["datetimeFrom"]["local"][:19]): h["value"] for h in results}   # [:19] drops "+05:30"
    s = pd.Series(values, dtype=float).sort_index()
    return s[(s >= VALID_PM25[0]) & (s <= VALID_PM25[1])]   # Drop impossible values


def fetch_station_pm25(hours: int = 96) -> pd.Series:
    """Hourly PM2.5 at Shadipur for the last `hours` hours. Picks whichever PM2.5 sensor actually has data
    (the station lists several, some of them dead)."""
    now = datetime.now(timezone.utc)
    with httpx.Client(timeout=TIMEOUT, headers=_openaq_headers()) as client:
        r = client.get(f"{OPENAQ}/locations/{LOCATION_ID}")
        r.raise_for_status()
        sensors = [s for s in r.json()["results"][0]["sensors"] if s["parameter"]["name"] == "pm25"]
        best = pd.Series(dtype=float)
        for s in sensors:                                # Try every PM2.5 sensor, keep the one with the most data
            r = client.get(f"{OPENAQ}/sensors/{s['id']}/hours", params={
                "datetime_from": (now - timedelta(hours=hours)).isoformat(), "datetime_to": now.isoformat(), "limit": 500})
            r.raise_for_status()
            data = _hours_to_series(r.json()["results"])
            if len(data) > len(best):
                best = data
    return best


def fetch_weather(past_days: int = 5, forecast_days: int = 2) -> pd.DataFrame:
    """Hourly weather (recent past + forecast) at P1's grid point, in local time, with P1's variables and units."""
    r = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=TIMEOUT, params={
        "latitude": WEATHER_POINT[0], "longitude": WEATHER_POINT[1], "hourly": ",".join(WEATHER_VARS),
        "past_days": past_days, "forecast_days": forecast_days, "timezone": "Asia/Kolkata"})
    r.raise_for_status()
    wx = pd.DataFrame(r.json()["hourly"]).set_index("time")
    wx.index = pd.to_datetime(wx.index)
    return wx


def fetch_cams(past_days: int = 1, forecast_days: int = 2) -> pd.Series:
    """CAMS (global model) PM2.5 estimate for the station's location: the fallback when the station feed is down."""
    r = httpx.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=TIMEOUT, params={
        "latitude": STATION_POINT[0], "longitude": STATION_POINT[1], "hourly": "pm2_5",
        "past_days": past_days, "forecast_days": forecast_days, "timezone": "Asia/Kolkata"})
    r.raise_for_status()
    data = r.json()["hourly"]
    return pd.Series(data["pm2_5"], index=pd.to_datetime(data["time"]), dtype=float).dropna()
