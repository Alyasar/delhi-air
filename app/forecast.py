"""The forecast: live data -> features -> model -> a plain-language answer (with a fallback if the station is down)."""

from datetime import datetime, timedelta                 # Times
from zoneinfo import ZoneInfo                            # Indian time, whatever the server's clock is set to
from pathlib import Path                                 # Model file location
import time                                              # Weather cache timing
import httpx                                             # To recognise HTTP errors in short messages
import joblib                                            # Load the trained model
import pandas as pd                                      # Tables

from app import data_sources                             # Live data (OpenAQ, Open-Meteo)
from app.features import build_features                  # Same feature code as training

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
MAX_DATA_AGE = timedelta(hours=6)                        # Older station data counts as "down" -> fallback
MIN_FORECAST_HOURS = 12                                  # Need at least this many forecast hours from the station model
WEATHER_CACHE_SECONDS = 2 * 60 * 60                      # The weather forecast changes slowly: fetch it at most every 2 hours
STATION_CACHE_SECONDS = 30 * 60                          # The station updates hourly: fetch it at most every 30 minutes

# India's National AQI categories for PM2.5 (µg/m³), with plain-language advice
CATEGORIES = [
    (30, "Good", "#00b050", "Air is good. Enjoy outdoor activities."),
    (60, "Satisfactory", "#92d050", "Air is acceptable. Very sensitive people may want to limit long outdoor exertion."),
    (90, "Moderate", "#ffd700", "People with asthma, heart or lung disease, children and older adults should reduce long outdoor exertion."),
    (120, "Poor", "#ff9900", "Avoid long or heavy outdoor exercise. Sensitive groups should stay indoors as much as possible."),
    (250, "Very Poor", "#ff0000", "Avoid outdoor activity. Wear an N95 mask outside and keep windows closed."),
    (float("inf"), "Severe", "#a00000", "Health emergency level. Stay indoors, use an air purifier if you can, and wear an N95 if you must go out."),
]

_bundle = None                                           # The loaded model (loaded once)
_weather = {"data": None, "time": 0.0}                   # Last weather download (fewer calls = fewer rate limits)
_station = {"data": None, "time": 0.0}                   # Last station download (shared by every request)
IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    """Current Indian time without a time zone label, like the station and weather data (cloud servers run on UTC)."""
    return datetime.now(IST).replace(tzinfo=None)


def category(pm25: float) -> dict:
    for limit, name, color, advice in CATEGORIES:
        if pm25 <= limit:
            return {"name": name, "color": color, "advice": advice}


def model_bundle() -> dict:
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)                # {"model", "features", "feature_set", "trained_until", ...}
    return _bundle


def cached_weather() -> pd.DataFrame:
    """The weather forecast, re-downloaded at most every 2 hours."""
    if _weather["data"] is None or time.time() - _weather["time"] > WEATHER_CACHE_SECONDS:
        _weather["data"], _weather["time"] = data_sources.fetch_weather(), time.time()
    return _weather["data"]


def cached_station_pm25() -> pd.Series:
    """The station's last 96 hours of PM2.5, re-downloaded at most every 30 minutes."""
    if _station["data"] is None or time.time() - _station["time"] > STATION_CACHE_SECONDS:
        _station["data"], _station["time"] = data_sources.fetch_station_pm25(hours=96), time.time()
    return _station["data"]


def short_reason(e: Exception) -> str:
    """A short, readable reason for the page (the full error goes to the server log)."""
    if isinstance(e, httpx.HTTPStatusError):
        return f"{e.request.url.host} answered HTTP {e.response.status_code}"
    if isinstance(e, httpx.HTTPError):
        return f"could not reach {getattr(e.request, 'url', 'a data source')}"
    return str(e)[:150]


def _points(series: pd.Series) -> list[dict]:
    return [{"time": t.to_pydatetime(), "pm25": round(float(v), 1)} for t, v in series.items()]


def station_forecast(weather: pd.DataFrame | None = None) -> dict:
    """Forecast for the next 24 hours from the station model. Raises if the live data isn't good enough.
    weather: hourly weather supplied by the caller (the web page fetches it from the visitor's browser);
    if None, the server downloads it itself."""
    bundle = model_bundle()
    pm25 = cached_station_pm25()                         # 48 h of lags for each of the last 24 input hours
    if pm25.empty:
        raise RuntimeError("the station returned no PM2.5 data")
    latest = pm25.index.max()
    if now_ist() - latest.to_pydatetime() > MAX_DATA_AGE:
        raise RuntimeError(f"the station's latest data ({latest:%d %b %H:%M}) is too old")
    weather = weather if weather is not None else cached_weather()
    feats = build_features(pm25, weather)
    rows = feats.loc[latest - timedelta(hours=23): latest, bundle["features"]].dropna()   # The last 24 input hours
    if len(rows) < MIN_FORECAST_HOURS:
        raise RuntimeError(f"only {len(rows)} complete input hours (need {MIN_FORECAST_HOURS})")
    forecast = pd.Series(bundle["model"].predict(rows), index=rows.index + timedelta(hours=24))   # Input hour + 24 h
    return {"source": "station_model", "data_until": latest.to_pydatetime(), "forecast": forecast.clip(lower=0),
            "observed": pm25.loc[latest - timedelta(hours=23):]}


def cams_forecast(reason: str) -> dict:
    """Fallback: the CAMS global-model estimate for the next 24 hours (clearly labelled as less accurate)."""
    cams = data_sources.fetch_cams()
    now = pd.Timestamp(now_ist()).floor("h")
    return {"source": "cams_fallback", "reason": reason, "data_until": None,
            "forecast": cams.loc[now: now + timedelta(hours=23)], "observed": pd.Series(dtype=float)}


def make_forecast(weather: pd.DataFrame | None = None) -> dict:
    """The full answer the API returns."""
    try:
        result = station_forecast(weather)
        note = "Forecast by our model from live measurements at the Shadipur station and the weather forecast."
    except Exception as e:                               # Station feed or weather down, stale or incomplete
        print(f"[fallback] station forecast failed: {e!r}", flush=True)   # Full detail in the server log
        result = cams_forecast(reason=short_reason(e))
        try:                                             # Still show the latest measurements if the station itself works
            pm25 = cached_station_pm25()
            result["observed"] = pm25.loc[pm25.index.max() - timedelta(hours=23):]
        except Exception:
            pass
        note = (f"Our forecast isn't available right now ({short_reason(e)}). Showing the CAMS global-model estimate "
                "instead, which is usually much less accurate for this location. Please check again later.")
    fc = result["forecast"]
    upcoming = fc[fc.index >= pd.Timestamp(now_ist()).floor("h")]   # Station data lags ~2-3 h, so the first hours are already past
    summary = upcoming if len(upcoming) >= MIN_FORECAST_HOURS else fc   # Headline numbers describe the hours still to come
    average, peak_time = float(summary.mean()), summary.idxmax()
    bundle = model_bundle()
    return {
        "source": result["source"], "note": note,
        "station": "Shadipur, Delhi (CPCB)",
        "data_until": result["data_until"], "generated_at": now_ist(),
        "average_pm25": round(average, 1), "peak_pm25": round(float(summary.max()), 1), "peak_time": peak_time.to_pydatetime(),
        "category": category(average),
        "forecast": _points(fc), "observed": _points(result["observed"]),
        "model": {"feature_set": bundle["feature_set"], "trained_until": bundle["trained_until"],
                  "test_metrics": bundle.get("test_metrics", {})},
    }
