"""FastAPI app: the forecast API, a health check, automatic docs, and the web page."""

import threading                                         # Lock around the cache
import time                                              # Cache timing
from datetime import datetime                            # Response times
from pathlib import Path                                 # File locations

from dotenv import load_dotenv                           # Local development: read .env (in Docker, env vars are passed in)
from fastapi import FastAPI, HTTPException               # The web framework
from fastapi.responses import FileResponse               # Serve the web page
from fastapi.staticfiles import StaticFiles              # Serve CSS/JS files
from pydantic import BaseModel, Field, model_validator  # Typed request/response models (become the API schema)
import pandas as pd                                      # Turn posted weather into a table

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from app.forecast import make_forecast, model_bundle     # noqa: E402  (after load_dotenv, so the key is available)

STATIC = Path(__file__).resolve().parent / "static"
CACHE_SECONDS = 30 * 60                                  # Recompute at most every 30 minutes (the station updates hourly)

app = FastAPI(title="Delhi Air: tomorrow's PM2.5", version="1.0.0",
              description="24-hour PM2.5 forecast for the Shadipur (CPCB) station in Delhi, from live measurements and the weather forecast.")


class HourPoint(BaseModel):
    time: datetime                                       # Local (Indian) time
    pm25: float                                          # µg/m³


class Category(BaseModel):
    name: str                                            # e.g. "Very Poor" (India's National AQI category)
    color: str                                           # Hex colour for the UI
    advice: str                                          # Plain-language health advice


class ModelInfo(BaseModel):
    feature_set: str
    trained_until: str
    test_metrics: dict


class Forecast(BaseModel):
    source: str                                          # "station_model" or "cams_fallback"
    note: str                                            # One sentence explaining where the forecast comes from
    station: str
    data_until: datetime | None                          # Latest station hour used (None for the fallback)
    generated_at: datetime
    average_pm25: float                                  # Average over the next 24 hours
    peak_pm25: float
    peak_time: datetime
    category: Category                                   # Category of the 24-hour average
    forecast: list[HourPoint]                            # Next 24 hours
    observed: list[HourPoint]                            # Last 24 measured hours (empty for the fallback)
    model: ModelInfo


class HourlyWeather(BaseModel):
    """Open-Meteo's "hourly" block, exactly as its API returns it (local Indian time)."""
    time: list[str] = Field(min_length=72, max_length=240)
    temperature_2m: list[float | None]
    relative_humidity_2m: list[float | None]
    dew_point_2m: list[float | None]
    precipitation: list[float | None]
    wind_speed_10m: list[float | None]
    wind_direction_10m: list[float | None]
    surface_pressure: list[float | None]
    cloud_cover: list[float | None]

    @model_validator(mode="after")
    def check(self):
        limits = {"temperature_2m": (-15, 55), "relative_humidity_2m": (0, 100), "dew_point_2m": (-40, 40),
                  "precipitation": (0, 300), "wind_speed_10m": (0, 250), "wind_direction_10m": (0, 360),
                  "surface_pressure": (850, 1100), "cloud_cover": (0, 100)}   # Physically plausible ranges for Delhi
        for name, (low, high) in limits.items():
            values = getattr(self, name)
            if len(values) != len(self.time):
                raise ValueError(f"{name} has {len(values)} values for {len(self.time)} hours")
            if any(v is not None and not low <= v <= high for v in values):
                raise ValueError(f"{name} has values outside {low}..{high}")
        return self


class WeatherInput(BaseModel):
    hourly: HourlyWeather                                # The page posts Open-Meteo's response as-is


_cache = {"value": None, "time": 0.0}
_lock = threading.Lock()


@app.get("/api/forecast", response_model=Forecast, summary="24-hour PM2.5 forecast")
def get_forecast():
    """The next 24 hours of PM2.5 at Shadipur, the category of the average and health advice. Cached for 30 minutes."""
    with _lock:                                          # One computation at a time; others reuse the cached answer
        if _cache["value"] is None or time.time() - _cache["time"] > CACHE_SECONDS:
            try:
                _cache["value"], _cache["time"] = make_forecast(), time.time()
            except Exception as e:                       # Even the fallback failed (e.g. no internet)
                if _cache["value"] is None:
                    raise HTTPException(status_code=503, detail=f"Forecast temporarily unavailable: {e}")
        return _cache["value"]                           # Serve the last good answer if a refresh failed


@app.post("/api/forecast", response_model=Forecast, summary="Forecast using weather supplied by the caller")
def post_forecast(body: WeatherInput):
    """Same forecast, but with weather the caller fetched from Open-Meteo itself (the web page does this from the
    visitor's browser, so shared cloud IPs don't hit Open-Meteo's rate limit). Not cached across visitors,
    so one caller's input can never change what others see. Station data is still fetched (and cached) by the server."""
    weather = pd.DataFrame(body.hourly.model_dump()).set_index("time")
    weather.index = pd.to_datetime(weather.index)
    try:
        return make_forecast(weather)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Forecast temporarily unavailable: {e}")


@app.get("/health", summary="Health check")
def health():
    """For the host's health checks: the app is up and the model is loaded."""
    return {"status": "ok", "model_trained_until": model_bundle()["trained_until"]}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")           # The product page


app.mount("/static", StaticFiles(directory=STATIC), name="static")
