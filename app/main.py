"""FastAPI app: the forecast API, a health check, automatic docs, and the web page."""

import threading                                         # Lock around the cache
import time                                              # Cache timing
from datetime import datetime                            # Response times
from pathlib import Path                                 # File locations

from dotenv import load_dotenv                           # Local development: read .env (in Docker, env vars are passed in)
from fastapi import FastAPI, HTTPException               # The web framework
from fastapi.responses import FileResponse               # Serve the web page
from fastapi.staticfiles import StaticFiles              # Serve CSS/JS files
from pydantic import BaseModel                           # Typed response models (become the API schema)

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


@app.get("/health", summary="Health check")
def health():
    """For the host's health checks: the app is up and the model is loaded."""
    return {"status": "ok", "model_trained_until": model_bundle()["trained_until"]}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")           # The product page


app.mount("/static", StaticFiles(directory=STATIC), name="static")
