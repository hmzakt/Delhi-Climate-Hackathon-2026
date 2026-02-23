"""
main.py — FastAPI server for Delhi Air Pollution Source Classifier

Run:   uvicorn main:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import json
from pathlib import Path
from datetime import datetime
import requests
from math import radians, sin, cos, sqrt, atan2

# Import LLM explanation functions
from llm_explainer import (
    prepare_pollution_context,
    get_authority_prompt,
    get_public_prompt,
    call_llm,
    get_fallback_explanation,
    GOOGLE_API_KEY
)

# LOAD MODEL + METADATA ON STARTUP
MODEL_DIR = Path("models")

with open(MODEL_DIR / "model_metadata.json") as f:
    META = json.load(f)

MODEL = xgb.XGBClassifier()
MODEL.load_model(str(MODEL_DIR / "source_classifier.json"))

CLASS_NAMES = META["class_names"]
FEATURE_COLS = META["feature_columns"]

# Station coordinates
STATIONS = {
    "R K Puram, Delhi - DPCC":    {"lat": 28.5633, "lon": 77.1869},
    "Punjabi Bagh, Delhi - DPCC": {"lat": 28.6740, "lon": 77.1310},
}

HIGHWAYS = [
    (28.565, 77.180), (28.560, 77.120),
    (28.700, 77.170), (28.620, 77.230),
]
INDUSTRIAL_ZONES = [
    (28.530, 77.270), (28.637, 77.142),
    (28.650, 77.290), (28.700, 77.100),
]
BRICK_KILNS = [
    (28.580, 77.050), (28.530, 77.350),
    (28.730, 77.050),
]

# ── Multi-year festive dates (must mirror data_pipeline.py)
FESTIVE_DATES_DIWALI = {
    "2024-10-30", "2024-10-31", "2024-11-01", "2024-11-02", "2024-11-03",
    "2025-10-18", "2025-10-19", "2025-10-20", "2025-10-21", "2025-10-22",
    "2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10",
}
FESTIVE_DATES_OTHER = {
    "2024-10-12", "2024-10-13",
    "2025-10-02", "2025-10-03",
    "2026-10-21", "2026-10-22",
    "2024-12-31", "2025-01-01",
    "2025-12-31", "2026-01-01",
    "2026-12-31", "2027-01-01",
    "2024-11-07", "2024-11-08",
    "2025-10-26", "2025-10-27",
    "2026-11-15", "2026-11-16",
}
ALL_FESTIVE_DATES = FESTIVE_DATES_DIWALI | FESTIVE_DATES_OTHER
DIWALI_DATES = FESTIVE_DATES_DIWALI 

# Pre-compute festive dates as date objects for proximity calculation
_FESTIVE_DATE_OBJS = sorted([datetime.strptime(d, "%Y-%m-%d").date() for d in ALL_FESTIVE_DATES])


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def min_dist(lat, lon, landmarks):
    return min(haversine_km(lat, lon, lm[0], lm[1]) for lm in landmarks)

# Global cache for API AQI to avoid redundant hits per request
aqi_cache = {}

def get_aqi_from_api(lat: float, lon: float, target_time: datetime):
    """Fetches US AQI from Open-Meteo Air Quality API."""
    cache_key = (round(lat, 2), round(lon, 2), target_time.strftime("%Y-%m-%d-%H"))
    if cache_key in aqi_cache:
        return aqi_cache[cache_key]

    date_str = target_time.strftime("%Y-%m-%d")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "us_aqi",
        "timezone": "GMT",
        "start_date": date_str,
        "end_date": date_str,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            # The API returns us_aqi as a list for the requested date range
            # We index it by the hour of the day (0-23)
            aqi_list = data.get("hourly", {}).get("us_aqi", [])
            if aqi_list:
                target_hour = target_time.hour
                aqi_val = aqi_list[target_hour]
                aqi_cache[cache_key] = aqi_val
                return aqi_val
        else:
            print(f"AQI API Error: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"AQI Fetch Exception for {lat}, {lon} at {target_time}: {e}")
    
    return 0 

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy (S)"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"



app = FastAPI(
    title="PIE — Pollution Intelligence Engine",
    description="Delhi Air Pollution Source Fingerprint API. Identifies the most likely source of air pollution from sensor + weather + fire data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class PollutionReading(BaseModel):
    """A single hourly reading from a monitoring station."""

    # Required
    pm25: float = Field(..., description="PM2.5 concentration (µg/m³)")
    temperature_2m: float = Field(..., description="Temperature at 2m (°C)")
    wind_speed_10m: float = Field(..., description="Wind speed at 10m (km/h)")
    wind_direction_10m: float = Field(..., description="Wind direction (degrees, 0-360)")
    relative_humidity_2m: float = Field(..., description="Relative humidity (%)")

    # Optional — will use defaults if missing
    surface_pressure: Optional[float] = Field(None, description="Surface pressure (hPa)")
    precipitation: Optional[float] = Field(0.0, description="Precipitation (mm)")
    cloud_cover: Optional[float] = Field(50.0, description="Cloud cover (%)")

    no2: Optional[float] = Field(None, description="NO2 concentration (µg/m³)")
    pm10: Optional[float] = Field(None, description="PM10 concentration (µg/m³)")

    fire_count_100km: Optional[int] = Field(0, description="Fires within 100km")
    total_frp_100km: Optional[float] = Field(0.0, description="Total FRP within 100km")
    max_frp_100km: Optional[float] = Field(0.0, description="Max FRP within 100km")
    upwind_fire_flag: Optional[int] = Field(0, description="1 if fire is upwind")
    fire_data_available: Optional[int] = Field(0, description="1 if satellite data exists")

    # Station location — defaults to R K Puram
    latitude: float = Field(28.5633, description="Station latitude")
    longitude: float = Field(77.1869, description="Station longitude")

    # Timestamp (ISO format) — defaults to now
    timestamp: Optional[str] = Field(None, description="ISO timestamp, e.g. 2026-02-20T08:30:00+05:30")

    # Lag features (optional — if you have previous readings)
    pm25_lag_1h: Optional[float] = None
    pm25_lag_3h: Optional[float] = None
    pm25_lag_6h: Optional[float] = None
    wind_speed_lag_3h: Optional[float] = None
    fire_frp_lag_3h: Optional[float] = None


class PredictionResponse(BaseModel):
    predicted_source: str
    confidence: float
    probabilities: dict[str, float]
    top_factors: list[str]
    reading_summary: dict
    model_predicted_source: Optional[str] = None
    refinement_applied: bool = False
    explanatory_tags: list[str] = Field(default_factory=list)


class BatchRequest(BaseModel):
    readings: list[PollutionReading]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: dict


class ExplanationRequest(BaseModel):
    station_name: str
    timestamp: str
    audience: str = Field(..., description="'public' or 'authority'")


class ExplanationResponse(BaseModel):
    explanation: str
    context: Dict[str, Any]
    audience: str
    model_used: str


class FeedbackRequest(BaseModel):
    explanation_id: Optional[str] = None
    helpful: bool
    feedback_text: Optional[str] = None

# ═════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (mirrors data_pipeline.py)
# ═════════════════════════════════════════════════════════════════

def build_features(reading: PollutionReading) -> pd.DataFrame:
    """Convert a PollutionReading into the 47-feature vector the model expects."""

    # Parse timestamp
    if reading.timestamp:
        dt = pd.Timestamp(reading.timestamp)
    else:
        dt = pd.Timestamp.now(tz="Asia/Kolkata")

    dt_ist = dt.tz_convert("Asia/Kolkata") if dt.tzinfo else dt.tz_localize("Asia/Kolkata")
    hour_ist = dt_ist.hour

    # Temporal
    day_of_week = dt_ist.dayofweek
    month = dt_ist.month
    is_weekend = int(day_of_week >= 5)
    is_night = int(hour_ist >= 20 or hour_ist <= 6)
    is_rush_hour = int((8 <= hour_ist <= 10) or (17 <= hour_ist <= 20))
    date_str = dt_ist.strftime("%Y-%m-%d")
    is_diwali = int(date_str in DIWALI_DATES)
    is_festive = int(date_str in ALL_FESTIVE_DATES)

    # festive_proximity: days to nearest festive date (capped at 15)
    query_date = dt_ist.date()
    festive_proximity = min(
        (min(abs((query_date - fd).days) for fd in _FESTIVE_DATE_OBJS) if _FESTIVE_DATE_OBJS else 15),
        15
    )

    # is_festive_evening: strong firecracker fingerprint
    is_festive_evening = int(
        festive_proximity <= 2 and
        (hour_ist >= 16 or hour_ist <= 3) and
        reading.pm25 > 120
    )

    if month in [11, 12, 1, 2]:
        season = 0
    elif month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8, 9]:
        season = 2
    else:
        season = 3

    # Meteorological
    ws = reading.wind_speed_10m
    temp = reading.temperature_2m
    rh = reading.relative_humidity_2m
    sp = reading.surface_pressure if reading.surface_pressure is not None else 1013.0
    precip = reading.precipitation if reading.precipitation is not None else 0.0
    cloud = reading.cloud_cover if reading.cloud_cover is not None else 50.0

    wind_speed_inv = 1.0 / (ws + 0.1)
    stagnation_index = wind_speed_inv * rh
    dispersion_index = ws * max(temp, 1.0)
    low_wind_flag = int(ws < 2.0)
    cold_night_flag = int(temp < 10.0 and is_night)
    precip_flag = int(precip > 0.1)

    # Fire
    fc = reading.fire_count_100km or 0
    frp = reading.total_frp_100km or 0.0
    mfrp = reading.max_frp_100km or 0.0
    upwind = reading.upwind_fire_flag or 0
    fire_avail = reading.fire_data_available or 0
    fire_x_low_wind = frp * wind_speed_inv

    # Chemical ratios
    no2 = reading.no2
    pm10 = reading.pm10
    pm25 = reading.pm25
    pm25_no2_ratio = pm25 / (no2 + 0.1) if no2 is not None else np.nan
    pm10_pm25_ratio = pm10 / (pm25 + 0.1) if pm10 is not None else np.nan
    coarse_fraction = ((pm10 - pm25) / (pm10 + 0.1)) if pm10 is not None else np.nan

    # PM2.5 derived
    pm25_wind_ratio = pm25 / (ws + 0.1)

    # Lag/rolling — use provided values when available.
    pm25_lag_1h = reading.pm25_lag_1h if reading.pm25_lag_1h is not None else pm25
    pm25_lag_3h = reading.pm25_lag_3h if reading.pm25_lag_3h is not None else pm25
    pm25_lag_6h = reading.pm25_lag_6h if reading.pm25_lag_6h is not None else pm25
    pm25_diff_1h = pm25 - pm25_lag_1h
    ws_lag_3h = reading.wind_speed_lag_3h if reading.wind_speed_lag_3h is not None else ws
    frp_lag = reading.fire_frp_lag_3h if reading.fire_frp_lag_3h is not None else frp

    lag_values = [v for v in [reading.pm25_lag_1h, reading.pm25_lag_3h, reading.pm25_lag_6h] if v is not None]
    if lag_values:
        baseline = float(np.mean(lag_values))
        pm25_rolling_3h = float(np.mean(lag_values[:1] + [pm25]))
        pm25_rolling_6h = float(np.mean(lag_values[:2] + [pm25]))
        pm25_rolling_12h = float(np.mean(lag_values + [pm25]))
        pm25_std_6h = float(np.std(lag_values + [pm25]))
    else:
        # Conservative baseline when no history is provided.
        baseline = max(45.0, pm25 * 0.65)
        pm25_rolling_3h = max(35.0, pm25 * 0.80)
        pm25_rolling_6h = max(40.0, pm25 * 0.72)
        pm25_rolling_12h = baseline
        pm25_std_6h = max(2.0, pm25 * 0.08)

    # pm25_spike_ratio: current vs 12h baseline
    pm25_spike_ratio = pm25 / (pm25_rolling_12h + 0.1)

    # Spatial
    lat, lon = reading.latitude, reading.longitude
    dist_highway = min_dist(lat, lon, HIGHWAYS)
    dist_industrial = min_dist(lat, lon, INDUSTRIAL_ZONES)
    dist_kilns = min_dist(lat, lon, BRICK_KILNS)

    # Build feature dict in exact column order
    features = {
        "pm25": pm25,
        "temperature_2m": temp,
        "relative_humidity_2m": rh,
        "wind_speed_10m": ws,
        "wind_direction_10m": reading.wind_direction_10m,
        "surface_pressure": sp,
        "precipitation": precip,
        "cloud_cover": cloud,
        "fire_count_100km": fc,
        "total_frp_100km": frp,
        "max_frp_100km": mfrp,
        "upwind_fire_flag": upwind,
        "fire_data_available": fire_avail,
        "no2": no2 if no2 is not None else np.nan,
        "pm10": pm10 if pm10 is not None else np.nan,
        "hour_ist": hour_ist,
        "day_of_week": day_of_week,
        "month": month,
        "day": dt_ist.day,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "is_rush_hour": is_rush_hour,
        "season": season,
        "is_diwali": is_diwali,
        "is_festive": is_festive,
        "festive_proximity": festive_proximity,
        "is_festive_evening": is_festive_evening,
        "wind_speed_inv": wind_speed_inv,
        "stagnation_index": stagnation_index,
        "dispersion_index": dispersion_index,
        "low_wind_flag": low_wind_flag,
        "cold_night_flag": cold_night_flag,
        "precip_flag": precip_flag,
        "fire_x_low_wind": fire_x_low_wind,
        "pm25_no2_ratio": pm25_no2_ratio,
        "pm10_pm25_ratio": pm10_pm25_ratio,
        "coarse_fraction": coarse_fraction,
        "pm25_wind_ratio": pm25_wind_ratio,
        "pm25_rolling_3h": pm25_rolling_3h,
        "pm25_rolling_6h": pm25_rolling_6h,
        "pm25_rolling_12h": pm25_rolling_12h,
        "pm25_std_6h": pm25_std_6h,
        "pm25_spike_ratio": pm25_spike_ratio,
        "pm25_lag_1h": pm25_lag_1h,
        "pm25_lag_3h": pm25_lag_3h,
        "pm25_lag_6h": pm25_lag_6h,
        "pm25_diff_1h": pm25_diff_1h,
        "wind_speed_lag_3h": ws_lag_3h,
        "fire_frp_lag_3h": frp_lag,
        "dist_to_highway": dist_highway,
        "dist_to_industrial": dist_industrial,
        "dist_to_kilns": dist_kilns,
    }

    return pd.DataFrame([features])[FEATURE_COLS]


GENERIC_SOURCES = {"background", "urban_mix"}


def rule_candidate_from_features(features: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Return a specific source candidate when strong physical signals are present."""
    tags: List[str] = []

    pm25 = float(features.get("pm25", 0) or 0)
    wind = float(features.get("wind_speed_10m", 0) or 0)
    rh = float(features.get("relative_humidity_2m", 0) or 0)
    temp = float(features.get("temperature_2m", 0) or 0)
    no2 = features.get("no2", np.nan)
    pm10_ratio = features.get("pm10_pm25_ratio", np.nan)
    stagnation = float(features.get("stagnation_index", 0) or 0)
    spike = float(features.get("pm25_spike_ratio", 1.0) or 1.0)
    festive = int(features.get("is_festive", 0) or 0)
    diwali = int(features.get("is_diwali", 0) or 0)
    festive_evening = int(features.get("is_festive_evening", 0) or 0)
    night = int(features.get("is_night", 0) or 0)
    rush = int(features.get("is_rush_hour", 0) or 0)
    month = int(features.get("month", 0) or 0)
    upwind_fire = int(features.get("upwind_fire_flag", 0) or 0)
    fire_count = float(features.get("fire_count_100km", 0) or 0)
    wind_dir = float(features.get("wind_direction_10m", 0) or 0)

    if (festive or diwali or festive_evening) and (pm25 > 130) and ((spike > 1.35) or (wind < 4.0)):
        tags.append("festive spike signature")
        return "firecrackers", tags

    if (upwind_fire == 1 or fire_count > 0) and pm25 > 90:
        tags.append("fire influence signature")
        return "biomass", tags

    if (
        month in [11, 12, 1, 2]
        and rh > 70
        and wind < 3.5
        and pm25 > 95
        and stagnation > 40
    ):
        tags.append("winter secondary aerosol signature")
        return "secondary_aerosols", tags

    if isinstance(pm10_ratio, (int, float, np.floating)) and np.isfinite(pm10_ratio):
        if pm10_ratio > 2.2 and rh < 45 and pm25 > 60:
            tags.append("coarse-particle dust signature")
            return "soil_road_dust", tags

    if month in [12, 1, 2] and temp < 10 and night == 1 and pm25 > 180:
        tags.append("winter inversion signature")
        return "winter_inversion", tags

    if wind < 2.3 and pm25 > 95 and (night == 1 or stagnation > 35):
        tags.append("low-wind stagnation signature")
        return "stagnation", tags

    if (
        rush == 1
        and isinstance(no2, (int, float, np.floating))
        and np.isfinite(no2)
        and no2 > 40
        and pm25 > 70
    ):
        tags.append("rush-hour NO2 traffic signature")
        return "traffic", tags

    if month in [10, 11] and (wind_dir >= 300 or wind_dir <= 30) and pm25 > 140:
        tags.append("northwest seasonal transport signature")
        return "biomass", tags

    return None, tags


def refine_prediction(
    probabilities: Dict[str, float],
    model_class: str,
    features: Dict[str, Any],
) -> Tuple[str, bool, List[str]]:
    """Bias generic predictions toward specific tags when evidence is present."""
    tags: List[str] = []
    final_class = model_class
    applied = False
    model_conf = float(probabilities.get(model_class, 0.0))

    rule_class, rule_tags = rule_candidate_from_features(features)
    tags.extend(rule_tags)

    if rule_class and rule_class != model_class:
        rule_conf = float(probabilities.get(rule_class, 0.0))
        if (model_class in GENERIC_SOURCES and (rule_conf >= 0.08 or model_conf < 0.65)) or (
            model_conf < 0.45 and rule_conf >= 0.12
        ):
            final_class = rule_class
            applied = True
            tags.append("rule-guided refinement")

    if final_class in GENERIC_SOURCES:
        ranked = sorted(
            [(cls, prob) for cls, prob in probabilities.items() if cls not in GENERIC_SOURCES],
            key=lambda x: x[1],
            reverse=True,
        )
        if ranked:
            alt_class, alt_prob = ranked[0]
            if alt_prob >= 0.22 and (probabilities[final_class] - alt_prob) <= 0.20:
                final_class = alt_class
                applied = True
                tags.append("generic-class uncertainty fallback")

    return final_class, applied, tags

# ═════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "PIE — Pollution Intelligence Engine",
        "version": "1.0.0",
        "model": "XGBoost Source Classifier",
        "classes": CLASS_NAMES,
        "features": len(FEATURE_COLS),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/model/info")
def model_info():
    return {
        "model_type": "XGBoost Multi-class Classifier",
        "classes": CLASS_NAMES,
        "n_features": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "cv_accuracy": round(
            np.mean([s["accuracy"] for s in META["cv_scores"]]), 4
        ),
        "cv_f1_weighted": round(
            np.mean([s["f1_weighted"] for s in META["cv_scores"]]), 4
        ),
        "xgb_params": META["xgb_params"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: PollutionReading):
    """Predict the most likely pollution source for a single reading."""
    try:
        X = build_features(reading)
        proba = MODEL.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        model_pred_class = CLASS_NAMES[pred_idx]
        confidence = float(proba[pred_idx])
        prob_raw = {cls: float(p) for cls, p in zip(CLASS_NAMES, proba)}
        probabilities = {cls: round(p, 4) for cls, p in prob_raw.items()}
        feature_values = X.iloc[0].to_dict()
        pred_class, refinement_applied, rule_tags = refine_prediction(
            prob_raw,
            model_pred_class,
            feature_values,
        )
        confidence = prob_raw.get(pred_class, confidence)

        # Top contributing factors (heuristic based on feature values)
        factors = []

        # Compute festive context from timestamp for factor reporting
        if reading.timestamp:
            _dt = pd.Timestamp(reading.timestamp)
        else:
            _dt = pd.Timestamp.now(tz="Asia/Kolkata")
        _dt_ist = _dt.tz_convert("Asia/Kolkata") if _dt.tzinfo else _dt.tz_localize("Asia/Kolkata")
        _date_str = _dt_ist.strftime("%Y-%m-%d")
        _is_festive = _date_str in ALL_FESTIVE_DATES
        _is_diwali = _date_str in DIWALI_DATES
        if reading.pm25 > 150:
            factors.append(f"Very high PM2.5 ({reading.pm25:.0f} µg/m³)")
        if reading.wind_speed_10m < 2:
            factors.append("Near-zero wind speed (stagnation)")
        if (reading.fire_count_100km or 0) > 0:
            factors.append(f"{reading.fire_count_100km} fires within 100km")
        if (reading.upwind_fire_flag or 0) == 1:
            factors.append("Fire detected upwind")
        if reading.temperature_2m < 10:
            factors.append(f"Cold conditions ({reading.temperature_2m:.1f}°C)")
        if reading.relative_humidity_2m > 80:
            factors.append(f"High humidity ({reading.relative_humidity_2m:.0f}%)")
        # Festive/firecracker context
        if _is_festive or _is_diwali:
            factors.append("Festive period (Diwali/festival — firecracker emissions likely)")
        factors.extend(rule_tags)
        if not factors:
            factors.append("Normal conditions")

        return PredictionResponse(
            predicted_source=pred_class,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            top_factors=factors,
            reading_summary={
                "pm25": reading.pm25,
                "temp": reading.temperature_2m,
                "wind": reading.wind_speed_10m,
                "humidity": reading.relative_humidity_2m,
            },
            model_predicted_source=model_pred_class,
            refinement_applied=refinement_applied,
            explanatory_tags=rule_tags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    """Predict sources for multiple readings at once."""
    if len(batch.readings) > 500:
        raise HTTPException(status_code=400, detail="Max 500 readings per batch")

    predictions = []
    source_counts = {cls: 0 for cls in CLASS_NAMES}

    for reading in batch.readings:
        result = predict(reading)
        predictions.append(result)
        source_counts[result.predicted_source] += 1

    dominant = max(source_counts, key=source_counts.get)

    return BatchResponse(
        predictions=predictions,
        summary={
            "total_readings": len(batch.readings),
            "source_distribution": source_counts,
            "dominant_source": dominant,
            "avg_pm25": round(np.mean([r.pm25 for r in batch.readings]), 1),
        },
    )


@app.get("/stations")
def get_stations():
    """List available monitoring stations with coordinates."""
    return {
        name: {
            "latitude": info["lat"],
            "longitude": info["lon"],
            "dist_to_highway_km": round(min_dist(info["lat"], info["lon"], HIGHWAYS), 2),
            "dist_to_industrial_km": round(min_dist(info["lat"], info["lon"], INDUSTRIAL_ZONES), 2),
            "dist_to_kilns_km": round(min_dist(info["lat"], info["lon"], BRICK_KILNS), 2),
        }
        for name, info in STATIONS.items()
    }


@app.get("/predict/latest")
def predict_latest():
    """Get latest prediction for each station from the dataset. Used by the map UI."""
    return get_predictions_for_time(None)


@app.get("/predict/history")
def predict_history(timestamp: str):
    """Get predictions for all stations at a specific historical ISO timestamp."""
    try:
        dt = pd.to_datetime(timestamp)
        return get_predictions_for_time(dt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {e}")


def get_predictions_for_time(target_dt: Optional[datetime]):
    """Helper to fetch data and predict for all stations at a specific hour."""
    data_path = Path("data/model_ready_delhi.csv")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(data_path)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])

    results = []
    
    # If no target_dt, find the overall latest hour in the dataset
    if target_dt is None:
        target_dt = df["datetime_hour"].max()
    else:
        # Floor target_dt to the hour to match dataset
        target_dt = pd.to_datetime(target_dt).floor("h")

    # Filter for that specific hour
    hour_data = df[df["datetime_hour"] == target_dt]
    
    if hour_data.empty:
        # Try to find the closest hour if exact match fails
        closest_idx = (df["datetime_hour"] - target_dt).abs().idxmin()
        target_dt = df.loc[closest_idx, "datetime_hour"]
        hour_data = df[df["datetime_hour"] == target_dt]

    for stn_name, stn_info in STATIONS.items():
        stn_row = hour_data[hour_data["location_name"] == stn_name]
        
        # If no data for this station at this hour, skip or find latest for this station
        if stn_row.empty:
            full_stn_data = df[df["location_name"] == stn_name].sort_values("datetime_hour")
            if full_stn_data.empty: continue
            latest = full_stn_data.iloc[-1]
        else:
            latest = stn_row.iloc[0]

        # Build feature vector from the row directly
        feat_vals = {}
        for col in FEATURE_COLS:
            val = latest.get(col, np.nan)
            feat_vals[col] = float(val) if pd.notna(val) else np.nan

        X = pd.DataFrame([feat_vals])[FEATURE_COLS]
        proba = MODEL.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        model_pred = CLASS_NAMES[pred_idx]
        prob_raw = {cls: float(p) for cls, p in zip(CLASS_NAMES, proba)}
        pred_class, refinement_applied, rule_tags = refine_prediction(
            prob_raw,
            model_pred,
            feat_vals,
        )
        confidence = prob_raw.get(pred_class, float(proba[pred_idx]))

        # Also get last 24h of data for the sparkline (relative to target_dt)
        stn_history = df[df["location_name"] == stn_name]
        past_24 = stn_history[stn_history["datetime_hour"] <= target_dt].tail(24)
        
        pm25_history = past_24["pm25"].tolist()
        time_labels = past_24["datetime_hour"].dt.strftime("%H:%M").tolist()

        pm25 = float(latest["pm25"])
        no2 = float(latest["no2"]) if "no2" in latest and pd.notna(latest["no2"]) else None
        
        # Use API-fetched AQI instead of local calculation
        aqi = get_aqi_from_api(stn_info["lat"], stn_info["lon"], target_dt)

        results.append({
            "station_name": stn_name,
            "latitude": stn_info["lat"],
            "longitude": stn_info["lon"],
            "predicted_source": pred_class,
            "model_predicted_source": model_pred,
            "refinement_applied": refinement_applied,
            "explanatory_tags": rule_tags,
            "confidence": round(float(confidence), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, proba)},
            "pm25": pm25,
            "no2": no2,
            "aqi": round(aqi, 0),
            "aqi_category": get_aqi_category(aqi),
            "temperature": float(latest.get("temperature_2m", 0)),
            "wind_speed": float(latest.get("wind_speed_10m", 0)),
            "wind_direction": float(latest.get("wind_direction_10m", 0)),
            "humidity": float(latest.get("relative_humidity_2m", 0)),
            "timestamp": str(latest["datetime_hour"]),
            "pm25_history": pm25_history,
            "time_labels": time_labels,
        })

    return {
        "timestamp": str(target_dt),
        "stations": results, 
        "model_accuracy": round(np.mean([s["accuracy"] for s in META["cv_scores"]]), 4),
        "available_range": {
            "start": str(df["datetime_hour"].min()),
            "end": str(df["datetime_hour"].max())
        }
    }


@app.post("/explain", response_model=ExplanationResponse)
def explain_pollution(request: ExplanationRequest):
    """
    Generate AI-powered explanation for a pollution spike.
    Supports two audiences: 'public' (citizens) and 'authority' (policymakers).
    """
    try:
        # Load data and find the specific reading
        data_path = Path("data/model_ready_delhi.csv")
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.read_csv(data_path)
        df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
        
        # Parse timestamp and find matching row
        target_dt = pd.to_datetime(request.timestamp).floor("h")
        station_row = df[
            (df["location_name"] == request.station_name) &
            (df["datetime_hour"] == target_dt)
        ]
        
        if station_row.empty:
            # Try to find closest match
            station_data = df[df["location_name"] == request.station_name]
            if station_data.empty:
                raise HTTPException(status_code=404, detail=f"Station '{request.station_name}' not found")
            closest_idx = (station_data["datetime_hour"] - target_dt).abs().idxmin()
            station_row = station_data.loc[[closest_idx]]
        
        row = station_row.iloc[0]
        
        # Prepare datetime with timezone handling
        dt_utc = pd.to_datetime(row["datetime_hour"])
        if dt_utc.tzinfo is None:
            dt_ist = dt_utc.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            dt_ist = dt_utc.tz_convert("Asia/Kolkata")
        
        # Extract all relevant fields with safe defaults
        context = prepare_pollution_context(
            station_name=request.station_name,
            datetime_ist=dt_ist,
            pm25=float(row["pm25"]),
            no2=float(row["no2"]) if pd.notna(row.get("no2")) and row.get("no2") != -999 else None,
            pm10=float(row["pm10"]) if pd.notna(row.get("pm10")) and row.get("pm10") != -999 else None,
            temperature=float(row["temperature_2m"]),
            humidity=float(row["relative_humidity_2m"]),
            wind_speed=float(row["wind_speed_10m"]),
            wind_direction=float(row["wind_direction_10m"]),
            surface_pressure=float(row["surface_pressure"]) if pd.notna(row.get("surface_pressure")) else None,
            precipitation=float(row["precipitation"]) if pd.notna(row.get("precipitation")) else None,
            cloud_cover=float(row["cloud_cover"]) if pd.notna(row.get("cloud_cover")) else None,
            fire_count=int(row.get("fire_count_100km", 0)),
            frp=float(row.get("total_frp_100km", 0)),
            max_frp=float(row.get("max_frp_100km", 0)),
            upwind_fire=int(row.get("upwind_fire_flag", 0)),
            is_diwali=int(row.get("is_diwali", 0)),
            month=int(row.get("month", dt_ist.month)),
            day=int(row.get("day", dt_ist.day)),
            day_of_week=int(row.get("day_of_week", dt_ist.dayofweek)),
            season=int(row.get("season", 0)),
            is_weekend=int(row.get("is_weekend", 0)),
            is_night=int(row.get("is_night", 0)),
            is_rush_hour=int(row.get("is_rush_hour", 0)),
            stagnation_index=float(row["stagnation_index"]) if pd.notna(row.get("stagnation_index")) else None,
            dispersion_index=float(row["dispersion_index"]) if pd.notna(row.get("dispersion_index")) else None,
            label_predicted=row.get("source_label", "unknown")
        )
        
        # Get appropriate prompt
        if request.audience.lower() == "authority":
            system_prompt, user_prompt = get_authority_prompt(context)
        else:  # public
            system_prompt, user_prompt = get_public_prompt(context)
        
        # Call LLM
        explanation = call_llm(system_prompt, user_prompt)
        
        if explanation is None:
            # Use fallback explanation from llm_explainer
            explanation = get_fallback_explanation(context)
        
        model_used = "google-gemini-2.5-flash" if GOOGLE_API_KEY else "fallback"
        
        return ExplanationResponse(
            explanation=explanation,
            context=context,
            audience=request.audience,
            model_used=model_used
        )
    
    except Exception as e:
        import traceback
        error_detail = f"Error generating explanation: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log full error for debugging
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")


@app.post("/explain/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback on AI explanations to improve the system.
    In production, this would be stored in a database.
    """
    # For now, just log it (in production, store in DB)
    print(f"Feedback received: helpful={feedback.helpful}, text={feedback.feedback_text}")
    
    return {
        "status": "received",
        "message": "Thank you for your feedback! This helps us improve explanations."
    }


