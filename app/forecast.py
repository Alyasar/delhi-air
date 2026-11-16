"""The forecast: live data -> features -> model -> a plain-language answer (with a fallback if the station is down)."""

from datetime import datetime, timedelta                 # Times
from zoneinfo import ZoneInfo                            # Indian time, whatever the server's clock is set to
from pathlib import Path                                 # Model file location
import joblib                                            # Load the trained model
import pandas as pd                                      # Tables

from app import data_sources                             # Live data (OpenAQ, Open-Meteo)
from app.features import build_features                  # Same feature code as training

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
MAX_DATA_AGE = timedelta(hours=6)                        # Older station data counts as "down" -> fallback
MIN_FORECAST_HOURS = 12                                  # Need at least this many forecast hours from the station model

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


def _points(series: pd.Series) -> list[dict]:
    return [{"time": t.to_pydatetime(), "pm25": round(float(v), 1)} for t, v in series.items()]


def station_forecast() -> dict:
    """Forecast for the next 24 hours from the station model. Raises if the live data isn't good enough."""
    bundle = model_bundle()
    pm25 = data_sources.fetch_station_pm25(hours=96)     # 48 h of lags for each of the last 24 input hours
    if pm25.empty:
        raise RuntimeError("the station returned no PM2.5 data")
    latest = pm25.index.max()
    if now_ist() - latest.to_pydatetime() > MAX_DATA_AGE:
        raise RuntimeError(f"the station's latest data ({latest:%d %b %H:%M}) is too old")
    weather = data_sources.fetch_weather()
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


def make_forecast() -> dict:
    """The full answer the API returns."""
    try:
        result = station_forecast()
        note = "Forecast by our model from live measurements at the Shadipur station and the weather forecast."
    except Exception as e:                               # Station feed down, stale or incomplete
        result = cams_forecast(reason=str(e))
        note = (f"The Shadipur station data isn't available right now ({e}). Showing the CAMS global-model estimate "
                "instead, which is usually much less accurate for this location.")
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
