"""
data_pipeline.py — Delhi Air Pollution: Complete Data Pipeline
Produces a model-ready dataframe for pollution source fingerprinting.

Usage:  python data_pipeline.py
Output: data/model_ready_delhi.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import json
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

# ═════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

STATIONS = {
    "R K Puram, Delhi - DPCC":    {"lat": 28.5633, "lon": 77.1869, "loc_id": 17},
    "Punjabi Bagh, Delhi - DPCC": {"lat": 28.6740, "lon": 77.1310, "loc_id": 50},
}

# Spatial landmarks (lat, lon, name)
HIGHWAYS = [
    (28.565, 77.180, "Ring Road"), (28.560, 77.120, "NH-8"),
    (28.700, 77.170, "GT Karnal Road"), (28.620, 77.230, "Mathura Road"),
]
INDUSTRIAL_ZONES = [
    (28.530, 77.270, "Okhla"), (28.637, 77.142, "Naraina"),
    (28.650, 77.290, "Patparganj"), (28.700, 77.100, "Wazirpur"),
]
BRICK_KILNS = [
    (28.580, 77.050, "Dwarka kilns"), (28.530, 77.350, "Noida kilns"),
    (28.730, 77.050, "Rohtak kilns"),
]

# ── Multi-year festive dates (firecrackers are common) ──
# Diwali (main night ± 2 days), Dussehra, Chhath Puja, New Year
FESTIVE_DATES_DIWALI = pd.to_datetime([
    # Diwali 2024: Nov 1
    "2024-10-30", "2024-10-31", "2024-11-01", "2024-11-02", "2024-11-03",
    # Diwali 2025: Oct 20
    "2025-10-18", "2025-10-19", "2025-10-20", "2025-10-21", "2025-10-22",
    # Diwali 2026: Nov 8
    "2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10",
])
FESTIVE_DATES_OTHER = pd.to_datetime([
    # Dussehra 2024, 2025, 2026
    "2024-10-12", "2024-10-13",
    "2025-10-02", "2025-10-03",
    "2026-10-21", "2026-10-22",
    # New Year
    "2024-12-31", "2025-01-01",
    "2025-12-31", "2026-01-01",
    "2026-12-31", "2027-01-01",
    # Chhath Puja (bonfires & firecrackers)
    "2024-11-07", "2024-11-08",
    "2025-10-26", "2025-10-27",
    "2026-11-15", "2026-11-16",
])
ALL_FESTIVE_DATES = pd.DatetimeIndex(list(FESTIVE_DATES_DIWALI) + list(FESTIVE_DATES_OTHER)).unique()
# Keep backward-compatible alias
DIWALI_DATES = FESTIVE_DATES_DIWALI

OPENAQ_KEY = os.environ.get("API_KEY", "")
FIRMS_KEY  = os.environ.get("FIRMS_MAP_KEY", "")
OPENAQ_HEADERS = {"X-API-Key": OPENAQ_KEY, "Accept": "application/json"} if OPENAQ_KEY else {}

# ═════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    x = sin(lon2-lon1) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(lon2-lon1)
    return (np.degrees(atan2(x, y)) + 360) % 360

def min_dist_to_landmarks(lat, lon, landmarks):
    return min(haversine_km(lat, lon, lm[0], lm[1]) for lm in landmarks)

def safe_get(url, params=None, headers=None, timeout=30, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            if attempt == retries - 1:
                print(f"   ⚠ Request failed: {e}")
                return None
            time.sleep(1)
    return None

def calculate_indian_aqi(pm25=None, no2=None):
    """
    Calculates Indian National AQI based on CPCB breakpoints.
    Formula: I = [(I_hi - I_lo)/(B_hi - B_lo)] * (C - B_lo) + I_lo
    """
    def get_sub_index(conf, breakpoints):
        if conf is None or np.isnan(conf): return 0
        for b_lo, b_hi, i_lo, i_hi in breakpoints:
            if b_lo <= conf <= b_hi:
                return ((i_hi - i_lo) / (b_hi - b_lo)) * (conf - b_lo) + i_lo
        # If exceeds max breakpoint
        max_b = breakpoints[-1]
        return max_b[3]

    # PM2.5 breakpoints: (B_lo, B_hi, I_lo, I_hi)
    pm25_bp = [
        (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)
    ]
    # NO2 breakpoints: (B_lo, B_hi, I_lo, I_hi)
    no2_bp = [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
        (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)
    ]

    si_pm25 = get_sub_index(pm25, pm25_bp)
    si_no2 = get_sub_index(no2, no2_bp)
    
    return max(si_pm25, si_no2)

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"

# ═════════════════════════════════════════════════════════════════
# STEP 1: LOAD PM2.5
# ═════════════════════════════════════════════════════════════════

def load_pm25():
    print("=" * 60)
    print("STEP 1: Loading PM2.5 data")
    print("=" * 60)

    csv_path = DATA_DIR / "delhi_pm25_openaq_v3_final.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"PM2.5 CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["datetime_hour"] = df["datetime_utc"].dt.floor("h")

    df = df[["location_name", "latitude", "longitude", "datetime_hour", "pm25"]].copy()
    # Average duplicates per station-hour
    df = df.groupby(["location_name", "latitude", "longitude", "datetime_hour"], as_index=False).agg(
        pm25=("pm25", "mean")
    )

    print(f"   ✓ {len(df)} hourly PM2.5 records")
    print(f"   Range: {df['datetime_hour'].min()} → {df['datetime_hour'].max()}")
    return df

# ═════════════════════════════════════════════════════════════════
# STEP 2: FETCH WEATHER (Open-Meteo)
# ═════════════════════════════════════════════════════════════════

def fetch_weather(pm25_start, pm25_end):
    print("\n" + "=" * 60)
    print("STEP 2: Fetching weather (Open-Meteo Archive + Forecast)")
    print("=" * 60)

    cache = DATA_DIR / "weather_full.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["datetime_hour"])
        df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], utc=True)
        print(f"   ✓ Cached: {len(df)} rows")
        return df

    hourly_vars = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation,cloud_cover"

    # Split: archive (historic) + forecast (recent)
    archive_end = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    forecast_start = archive_end
    forecast_end = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    archive_start = pm25_start.strftime("%Y-%m-%d")

    all_dfs = []
    for stn_name, info in STATIONS.items():
        print(f"\n   📍 {stn_name}")
        lat, lon = info["lat"], info["lon"]
        parts = []

        # Archive
        print(f"   → Archive: {archive_start} to {archive_end}")
        r = safe_get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": lat, "longitude": lon,
            "start_date": archive_start, "end_date": archive_end,
            "hourly": hourly_vars, "timezone": "UTC",
        }, timeout=120)
        if r:
            data = r.json()
            if "hourly" in data:
                df_a = pd.DataFrame(data["hourly"])
                df_a.rename(columns={"time": "datetime_hour"}, inplace=True)
                parts.append(df_a)
                print(f"     ✓ {len(df_a)} rows")
            else:
                print(f"     ⚠ No data: {data.get('reason', 'unknown')}")

        # Forecast (recent days)
        print(f"   → Forecast: {forecast_start} to {forecast_end}")
        r2 = safe_get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "start_date": forecast_start, "end_date": forecast_end,
            "hourly": hourly_vars, "timezone": "UTC",
        }, timeout=60)
        if r2:
            data2 = r2.json()
            if "hourly" in data2:
                df_f = pd.DataFrame(data2["hourly"])
                df_f.rename(columns={"time": "datetime_hour"}, inplace=True)
                parts.append(df_f)
                print(f"     ✓ {len(df_f)} rows")

        if parts:
            df_c = pd.concat(parts, ignore_index=True)
            df_c["datetime_hour"] = pd.to_datetime(df_c["datetime_hour"])
            df_c = df_c.drop_duplicates("datetime_hour").sort_values("datetime_hour")
            df_c["location_name"] = stn_name
            all_dfs.append(df_c)

    if not all_dfs:
        raise RuntimeError("Weather fetch failed for all stations!")

    df_weather = pd.concat(all_dfs, ignore_index=True)
    df_weather["datetime_hour"] = pd.to_datetime(df_weather["datetime_hour"], utc=True)
    df_weather.to_csv(cache, index=False)
    print(f"\n   ✓ Total weather: {len(df_weather)} rows")
    return df_weather

# ═════════════════════════════════════════════════════════════════
# STEP 3: FIRE DATA (FIRMS)
# ═════════════════════════════════════════════════════════════════

def load_fire_data():
    print("\n" + "=" * 60)
    print("STEP 3: Loading fire data (FIRMS)")
    print("=" * 60)

    dfs = []
    # Load existing
    existing = DATA_DIR / "firms_area_fires_bbox.csv"
    if existing.exists():
        df_e = pd.read_csv(existing)
        dfs.append(df_e)
        print(f"   ✓ Existing: {len(df_e)} rows")

    # Try fetching 10 days
    if FIRMS_KEY:
        print("   → Fetching 10-day FIRMS data...")
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/VIIRS_SNPP_NRT/74.0,27.0,79.5,31.5/10"
        r = safe_get(url, timeout=60)
        if r and len(r.text) > 200:
            df_n = pd.read_csv(StringIO(r.text))
            dfs.append(df_n)
            print(f"     ✓ Fetched {len(df_n)} rows")

    if not dfs:
        print("   ⚠ No fire data — fire features will be zero")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Build acq_datetime
    if "acq_datetime" not in df.columns and {"acq_date", "acq_time"}.issubset(df.columns):
        df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
        df["acq_datetime"] = pd.to_datetime(df["acq_date"] + " " + df["acq_time"],
                                            format="%Y-%m-%d %H%M", errors="coerce")
    df["acq_datetime"] = pd.to_datetime(df["acq_datetime"], utc=True, errors="coerce")
    df["datetime_hour"] = df["acq_datetime"].dt.floor("h")
    df["frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)
    df = df.dropna(subset=["datetime_hour", "latitude", "longitude"])
    df = df.drop_duplicates(subset=["latitude", "longitude", "datetime_hour"])

    print(f"   ✓ Unique fire detections: {len(df)}")
    return df[["latitude", "longitude", "datetime_hour", "frp"]].copy()

def aggregate_fire_features(df_pm25, df_fire):
    """Compute per-station-hour fire proximity features without row explosion."""
    print("   Computing fire proximity features...")

    # Initialize
    n = len(df_pm25)
    fire_count   = np.zeros(n, dtype=int)
    total_frp    = np.zeros(n, dtype=float)
    max_frp      = np.zeros(n, dtype=float)
    upwind_fire  = np.zeros(n, dtype=int)
    fire_avail   = np.zeros(n, dtype=int)

    if df_fire.empty:
        df_pm25["fire_count_100km"] = fire_count
        df_pm25["total_frp_100km"]  = total_frp
        df_pm25["max_frp_100km"]    = max_frp
        df_pm25["upwind_fire_flag"] = upwind_fire
        df_pm25["fire_data_available"] = fire_avail
        return df_pm25

    fire_dates = set(df_fire["datetime_hour"].dt.normalize().unique())
    fire_hours_set = set(df_fire["datetime_hour"].unique())

    # Precompute: for each station, distance to every fire
    station_fire_dists = {}
    for stn_name, stn_info in STATIONS.items():
        dists = df_fire.apply(
            lambda r: haversine_km(stn_info["lat"], stn_info["lon"], r["latitude"], r["longitude"]),
            axis=1
        ).values
        station_fire_dists[stn_name] = dists

    # Process only hours that have fire data
    for idx in range(n):
        row_hour = df_pm25.iloc[idx]["datetime_hour"]
        row_stn  = df_pm25.iloc[idx]["location_name"]
        row_date = row_hour.normalize()

        if row_date in fire_dates:
            fire_avail[idx] = 1

        if row_hour not in fire_hours_set:
            continue

        # Fires in this hour
        hour_mask = df_fire["datetime_hour"] == row_hour
        dists = station_fire_dists[row_stn][hour_mask.values]
        frps  = df_fire.loc[hour_mask, "frp"].values

        nearby = dists <= 100
        if nearby.any():
            fire_count[idx] = nearby.sum()
            total_frp[idx]  = frps[nearby].sum()
            max_frp[idx]    = frps[nearby].max()

            # Upwind check
            wind_dir = df_pm25.iloc[idx].get("wind_direction_10m", np.nan)
            if not np.isnan(wind_dir):
                stn_lat = STATIONS[row_stn]["lat"]
                stn_lon = STATIONS[row_stn]["lon"]
                fire_lats = df_fire.loc[hour_mask, "latitude"].values[nearby]
                fire_lons = df_fire.loc[hour_mask, "longitude"].values[nearby]
                for fl, flo in zip(fire_lats, fire_lons):
                    fb = bearing_deg(stn_lat, stn_lon, fl, flo)
                    diff = abs((fb - wind_dir + 180) % 360 - 180)
                    if diff <= 60:
                        upwind_fire[idx] = 1
                        break

    df_pm25["fire_count_100km"]    = fire_count
    df_pm25["total_frp_100km"]     = total_frp
    df_pm25["max_frp_100km"]       = max_frp
    df_pm25["upwind_fire_flag"]    = upwind_fire
    df_pm25["fire_data_available"] = fire_avail
    print(f"   ✓ Fire features computed ({fire_count.sum()} nearby-fire events)")
    return df_pm25

# ═════════════════════════════════════════════════════════════════
# STEP 4: TRY NO2 / PM10 FROM OPENAQ
# ═════════════════════════════════════════════════════════════════

def try_fetch_openaq_pollutant(location_id, param_name, param_ids):
    """Try fetching a pollutant from OpenAQ v3. Returns DataFrame or None."""
    base = "https://api.openaq.org/v3"

    # 1. Find sensor
    r = safe_get(f"{base}/locations/{location_id}/sensors",
                 headers=OPENAQ_HEADERS, params={"limit": 100})
    if not r:
        return None

    sensors = r.json().get("results", [])
    target_sensor = None
    for s in sensors:
        p = s.get("parameter", {})
        if p.get("name", "").lower() == param_name or p.get("id") in param_ids:
            last = s.get("datetimeLast", {}).get("utc", "")
            if last and last > "2025-01-01":
                target_sensor = s
                break

    if not target_sensor:
        return None

    sensor_id = target_sensor["id"]
    print(f"     Found sensor {sensor_id} for {param_name}")

    # 2. Fetch hourly data (paginated)
    all_data = []
    for page in range(1, 40):
        r = safe_get(f"{base}/sensors/{sensor_id}/hours",
                     headers=OPENAQ_HEADERS,
                     params={"limit": 1000, "page": page})
        if not r:
            break
        results = r.json().get("results", [])
        if not results:
            break
        all_data.extend(results)
        if len(results) < 1000:
            break
        time.sleep(0.3)

    if not all_data:
        return None

    rows = []
    for h in all_data:
        dt = None
        if "period" in h and isinstance(h["period"], dict):
            df_from = h["period"].get("datetimeFrom", {})
            dt = df_from.get("utc") if isinstance(df_from, dict) else df_from
        elif "datetime" in h:
            dt = h["datetime"].get("utc") if isinstance(h["datetime"], dict) else h["datetime"]
        rows.append({"datetime_utc": dt, param_name: h.get("value")})

    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df["datetime_hour"] = df["datetime_utc"].dt.floor("h")
    df = df.dropna(subset=[param_name, "datetime_hour"])
    return df[["datetime_hour", param_name]]


def fetch_extra_pollutants():
    print("\n" + "=" * 60)
    print("STEP 4: Trying NO2 & PM10 from OpenAQ (optional)")
    print("=" * 60)

    if not OPENAQ_KEY:
        print("   ⚠ No OpenAQ API key — skipping")
        return {}

    results = {}
    # NO2 param_id=7, PM10 param_id=1
    for stn_name, info in STATIONS.items():
        loc_id = info["loc_id"]
        print(f"\n   📍 {stn_name} (ID: {loc_id})")

        for param_name, param_ids in [("no2", [7, 19]), ("pm10", [1])]:
            print(f"   → Trying {param_name}...")
            try:
                df = try_fetch_openaq_pollutant(loc_id, param_name, param_ids)
                if df is not None and len(df) > 100:
                    key = (stn_name, param_name)
                    results[key] = df
                    print(f"     ✓ Got {len(df)} {param_name} records")
                else:
                    print(f"     ⚠ Insufficient {param_name} data")
            except Exception as e:
                print(f"     ⚠ Error fetching {param_name}: {e}")

    return results

# ═════════════════════════════════════════════════════════════════
# STEP 5: MERGE ALL DATA
# ═════════════════════════════════════════════════════════════════

def merge_all(df_pm25, df_weather, df_fire, extra_pollutants):
    print("\n" + "=" * 60)
    print("STEP 5: Merging all datasets")
    print("=" * 60)

    # Merge PM2.5 + Weather
    weather_cols = ["datetime_hour", "location_name", "temperature_2m",
                    "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
                    "surface_pressure", "precipitation", "cloud_cover"]
    avail_cols = [c for c in weather_cols if c in df_weather.columns]
    df = pd.merge(df_pm25, df_weather[avail_cols], on=["datetime_hour", "location_name"], how="left")
    print(f"   ✓ PM2.5 + Weather: {len(df)} rows, {df['wind_speed_10m'].notna().sum()} with weather")

    # Merge fire features
    df = aggregate_fire_features(df, df_fire)

    # Merge extra pollutants (NO2, PM10)
    for (stn_name, param_name), df_poll in extra_pollutants.items():
        # De-duplicate pollutant data per hour
        df_poll_dedup = df_poll.groupby("datetime_hour", as_index=False).agg(
            **{param_name: (param_name, "mean")}
        )
        # Create a lookup dict: datetime_hour -> value
        lookup = dict(zip(df_poll_dedup["datetime_hour"], df_poll_dedup[param_name]))

        if param_name not in df.columns:
            df[param_name] = np.nan

        mask = df["location_name"] == stn_name
        df.loc[mask, param_name] = df.loc[mask, "datetime_hour"].map(lookup)
        cnt = df.loc[mask, param_name].notna().sum()
        print(f"   ✓ Merged {param_name} for {stn_name}: {cnt} values")

    print(f"   ✓ Merged shape: {df.shape}")
    return df

# ═════════════════════════════════════════════════════════════════
# STEP 6: FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════

def engineer_features(df):
    print("\n" + "=" * 60)
    print("STEP 6: Feature engineering")
    print("=" * 60)

    # ── IST time (UTC + 5:30) ──
    df["datetime_ist"] = df["datetime_hour"] + pd.Timedelta(hours=5, minutes=30)
    df["hour_ist"] = df["datetime_ist"].dt.hour
    df["day_of_week"] = df["datetime_ist"].dt.dayofweek
    df["month"] = df["datetime_ist"].dt.month
    df["day"] = df["datetime_ist"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_ist"] >= 20) | (df["hour_ist"] <= 6)).astype(int)
    df["is_rush_hour"] = (
        ((df["hour_ist"] >= 8) & (df["hour_ist"] <= 10)) |
        ((df["hour_ist"] >= 17) & (df["hour_ist"] <= 20))
    ).astype(int)

    # Season: 0=winter(Nov-Feb), 1=spring(Mar-May), 2=monsoon(Jun-Sep), 3=post_monsoon(Oct)
    def get_season(m):
        if m in [11, 12, 1, 2]: return 0
        elif m in [3, 4, 5]: return 1
        elif m in [6, 7, 8, 9]: return 2
        else: return 3
    df["season"] = df["month"].apply(get_season)

    # ── Festive / Diwali features ──
    df["date"] = df["datetime_ist"].dt.tz_localize(None).dt.normalize()
    df["is_diwali"] = df["date"].isin(pd.to_datetime(list(DIWALI_DATES))).astype(int)
    df["is_festive"] = df["date"].isin(pd.to_datetime(list(ALL_FESTIVE_DATES))).astype(int)

    # festive_proximity: days to nearest festive date (0 = on the day, capped at 15)
    all_fest_np = pd.to_datetime(list(ALL_FESTIVE_DATES)).values.astype('datetime64[ns]')
    def _days_to_nearest_fest(d):
        if pd.isna(d):
            return 15.0
        d_np = np.datetime64(d)
        diffs = np.abs((all_fest_np - d_np) / np.timedelta64(1, 'D'))
        return min(float(np.min(diffs)), 15.0)
    df["festive_proximity"] = df["date"].apply(_days_to_nearest_fest)

    # pm25_spike_ratio: current PM2.5 vs 12h rolling mean (firecrackers = sharp spike)
    # Will be calculated after rolling features are available; placeholder here
    # (computed below after rolling features)

    # is_festive_evening: strong firecracker fingerprint
    df["is_festive_evening"] = (
        (df["festive_proximity"] <= 2) &
        ((df["hour_ist"] >= 16) | (df["hour_ist"] <= 3)) &
        (df["pm25"] > 120)
    ).astype(int)

    df.drop(columns=["date", "datetime_ist"], inplace=True)

    print("   ✓ Temporal features (8)")

    # ── Meteorological features ──
    ws = df["wind_speed_10m"].fillna(df["wind_speed_10m"].median())
    temp = df["temperature_2m"].fillna(df["temperature_2m"].median())
    rh = df["relative_humidity_2m"].fillna(df["relative_humidity_2m"].median())

    df["wind_speed_inv"] = 1.0 / (ws + 0.1)
    df["stagnation_index"] = df["wind_speed_inv"] * rh
    df["dispersion_index"] = ws * np.maximum(temp, 1.0)
    df["low_wind_flag"] = (ws < 2.0).astype(int)
    df["cold_night_flag"] = ((temp < 10.0) & (df["is_night"] == 1)).astype(int)
    df["precip_flag"] = (df["precipitation"].fillna(0) > 0.1).astype(int)

    print("   ✓ Meteorological interaction features (6)")

    # ── Fire interaction ──
    df["fire_x_low_wind"] = df["total_frp_100km"] * df["wind_speed_inv"]
    print("   ✓ Fire interaction features (1)")

    # ── Chemical ratios (if available) ──
    if "no2" in df.columns:
        df["pm25_no2_ratio"] = df["pm25"] / (df["no2"].fillna(0) + 0.1)
    if "pm10" in df.columns:
        df["pm10_pm25_ratio"] = df["pm10"].fillna(0) / (df["pm25"] + 0.1)
        df["coarse_fraction"] = (df["pm10"].fillna(0) - df["pm25"]) / (df["pm10"].fillna(0) + 0.1)
    print("   ✓ Chemical ratio features (if data available)")

    # ── PM2.5 derived features ──
    df["pm25_wind_ratio"] = df["pm25"] / (ws + 0.1)

    # Sort by station + time for rolling/lag
    df = df.sort_values(["location_name", "datetime_hour"]).reset_index(drop=True)

    for stn in df["location_name"].unique():
        mask = df["location_name"] == stn
        pm = df.loc[mask, "pm25"]

        df.loc[mask, "pm25_rolling_3h"]  = pm.rolling(3, min_periods=1).mean()
        df.loc[mask, "pm25_rolling_6h"]  = pm.rolling(6, min_periods=1).mean()
        df.loc[mask, "pm25_rolling_12h"] = pm.rolling(12, min_periods=1).mean()
        df.loc[mask, "pm25_std_6h"]      = pm.rolling(6, min_periods=1).std().fillna(0)

        df.loc[mask, "pm25_lag_1h"]  = pm.shift(1)
        df.loc[mask, "pm25_lag_3h"]  = pm.shift(3)
        df.loc[mask, "pm25_lag_6h"]  = pm.shift(6)
        df.loc[mask, "pm25_diff_1h"] = pm.diff(1)

        ws_stn = df.loc[mask, "wind_speed_10m"]
        df.loc[mask, "wind_speed_lag_3h"] = ws_stn.shift(3)

        frp_stn = df.loc[mask, "total_frp_100km"]
        df.loc[mask, "fire_frp_lag_3h"] = frp_stn.shift(3)

    print("   ✓ Rolling / lag features (10)")

    # ── PM2.5 spike ratio (computed after rolling features) ──
    df["pm25_spike_ratio"] = df["pm25"] / (df["pm25_rolling_12h"].replace(0, 1) + 0.1)
    print("   ✓ Festive-awareness features (4): is_festive, festive_proximity, is_festive_evening, pm25_spike_ratio")

    # ── Spatial features (static per station) ──
    df["dist_to_highway"]    = df.apply(lambda r: min_dist_to_landmarks(r["latitude"], r["longitude"], HIGHWAYS), axis=1)
    df["dist_to_industrial"] = df.apply(lambda r: min_dist_to_landmarks(r["latitude"], r["longitude"], INDUSTRIAL_ZONES), axis=1)
    df["dist_to_kilns"]      = df.apply(lambda r: min_dist_to_landmarks(r["latitude"], r["longitude"], BRICK_KILNS), axis=1)
    print("   ✓ Spatial distance features (3)")

    print(f"\n   ✓ Total columns: {len(df.columns)}")
    return df

# ═════════════════════════════════════════════════════════════════
# STEP 7: GENERATE SOURCE LABELS (RULE-BASED)
# ═════════════════════════════════════════════════════════════════

def generate_source_labels(df):
    print("\n" + "=" * 60)
    print("STEP 7: Generating source labels (rule-based heuristic)")
    print("=" * 60)

    pm_median = df["pm25"].median()
    pm_75 = df["pm25"].quantile(0.75)

    # Default to urban_mix so unexplained cases don't collapse into background.
    labels = pd.Series("urban_mix", index=df.index)

    # Clean air: Truly low pollution
    labels[df["pm25"] < 30] = "clean_air"

    # Background: only genuinely low-impact baseline conditions.
    background_mask = (
        (df["pm25"] < 55) &
        (df["wind_speed_10m"] >= 3.0) &
        (df["relative_humidity_2m"] < 80) &
        (df["is_rush_hour"] == 0) &
        (df["is_festive"] == 0) &
        (df["fire_count_100km"] == 0) &
        (df["upwind_fire_flag"] == 0)
    )
    labels[background_mask] = "background"

    # ═══ FIRECRACKERS — Define mask FIRST to use as exclusion guard ═══
    festive_flag = (
        (df["is_festive"] == 1) |
        (df["is_diwali"] == 1) |
        (df["festive_proximity"] <= 1)
    )
    firecrackers_rule = (
        (festive_flag) &
        (df["pm25"] > 150) &
        ((df["hour_ist"] >= 16) | (df["hour_ist"] <= 4)) &
        (df["wind_speed_10m"] < 4.0)
    )
    firecrackers_spike = (
        (festive_flag) &
        (df["pm25"] > 120) &
        (df["pm25_spike_ratio"] > 1.5) &
        ((df["hour_ist"] >= 16) | (df["hour_ist"] <= 4))
    )
    firecrackers_mask = firecrackers_rule | firecrackers_spike

    # Secondary Aerosols: High Humidity + Winter + Calm + High PM2.5
    secondary = (
        (df["month"].isin([11, 12, 1, 2])) &
        (df["relative_humidity_2m"] > 70) &
        (df["wind_speed_10m"] < 4.0) &
        (df["pm25"] > 100) &
        (df["stagnation_index"] > 50) &
        (~firecrackers_mask) # Guard
    )
    labels[secondary] = "secondary_aerosols"

    # Soil & Road Dust
    if "pm10_pm25_ratio" in df.columns:
        dust = (
            (df["pm10_pm25_ratio"] > 2.2) &
            (df["relative_humidity_2m"] < 40) &
            (df["pm25"] > pm_median) &
            (~firecrackers_mask) # Guard
        )
        labels[dust] = "soil_road_dust"
    else:
        dust = pd.Series(False, index=df.index)

    # Stagnation / meteorological trapping
    stagnation = (
        (df["wind_speed_10m"] < 2.5) & 
        ((df["is_night"] == 1) | (df["hour_ist"].isin([7, 8, 9]))) & 
        (df["pm25"] > 100) &
        (~secondary) &
        (~firecrackers_mask) # Guard: prioritize firecrackers on festive nights
    )
    labels[stagnation] = "stagnation"

    # Traffic
    traffic = (
        (df["is_rush_hour"] == 1) &
        (df["pm25"] > pm_75) &
        (df["no2"].fillna(0) > 40) &
        (~secondary) & (~dust) & (~stagnation) &
        (~firecrackers_mask) # Guard
    )
    labels[traffic] = "traffic"

    # Now apply the firecrackers label (will overwrite background/urban_mix)
    labels[firecrackers_mask] = "firecrackers"

    # Biomass / stubble / waste burning
    nw_wind_bias = (
        (df["month"].isin([10, 11])) &
        ((df["wind_direction_10m"] >= 300) | (df["wind_direction_10m"] <= 30)) &
        (df["pm25"] > 150) &
        (~firecrackers_mask)
    )
    biomass_fire = (df["upwind_fire_flag"] == 1) & (~firecrackers_mask)
    labels[nw_wind_bias | biomass_fire] = "biomass"

    # Winter inversion
    inversion = (
        (df["month"].isin([12, 1, 2])) &
        (df["cold_night_flag"] == 1) &
        (df["pm25"] > 200) &
        (~secondary) & (~stagnation) &
        (~firecrackers_mask)
    )
    labels[inversion] = "winter_inversion"

    df["source_label"] = labels

    print("   Label distribution:")
    for lbl, cnt in df["source_label"].value_counts().items():
        print(f"     {lbl:20s}: {cnt:6d} ({100*cnt/len(df):.1f}%)")

    return df


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def augment_minority_class(df, minority_class="firecrackers", target_count=300):
    print(f"\n   → Augmenting '{minority_class}' class to {target_count} samples...")
    # Use as copy
    current_df = df[df["source_label"] == minority_class].copy()
    num_existing = len(current_df)
    
    if num_existing == 0:
        # Fallback: if no firecrackers exist in the rule-based labeling, 
        # use stagnation as template (since firecrackers have similar met fingerprint)
        print(f"   ! No samples found for {minority_class}. Using 'stagnation' as template...")
        current_df = df[df["source_label"] == "stagnation"].head(100).copy()
        num_existing = len(current_df)

    if num_existing == 0:
        print("   ! ERROR: No template samples found to augment.")
        return df

    num_to_add = target_count - num_existing
    if num_to_add <= 0:
        print(f"   ✓ Already has {num_existing} samples.")
        return df

    # ── Synthetic "Season Re-seasoning" ──
    # Since our training data is only January, we MUST synthesize October/Diwali/Festive samples
    # to teach the model to generalize for future Diwali predictions.
    
    new_samples = current_df.sample(num_to_add, replace=True).copy()
    
    # Add noise to continuous columns
    noise_cols = [
        "pm25", "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "stagnation_index", "pm25_spike_ratio", "pm25_rolling_3h"
    ]
    
    rng = np.random.RandomState(42)
    for col in noise_cols:
        if col in new_samples.columns:
            new_samples[col] = new_samples[col] * rng.uniform(0.95, 1.05, size=len(new_samples))
            new_samples[col] = new_samples[col].clip(lower=0)
            
    # Season Re-seasoning splits using simple array indexing
    n = len(new_samples)
    idx1, idx2 = n // 3, 2 * n // 3
    
    # 1. Diwali Synthetics (FORCE October context)
    diwali_syn = new_samples.iloc[:idx1].copy()
    if not diwali_syn.empty:
        diwali_syn["month"] = 10
        diwali_syn["season"] = 3 # post-monsoon
        diwali_syn["is_diwali"] = 1
        diwali_syn["is_festive"] = 1
        diwali_syn["festive_proximity"] = 0
        diwali_syn["is_festive_evening"] = 1
        diwali_syn["pm25"] = diwali_syn["pm25"].clip(lower=200)
        diwali_syn["pm25_spike_ratio"] = diwali_syn["pm25_spike_ratio"].clip(lower=2.0)
    
    # 2. New Year Synthetics (stays in month 1)
    newyear_syn = new_samples.iloc[idx1:idx2].copy()
    if not newyear_syn.empty:
        newyear_syn["month"] = 1
        newyear_syn["season"] = 0 # winter
        newyear_syn["is_diwali"] = 0
        newyear_syn["is_festive"] = 1
        newyear_syn["festive_proximity"] = 0
        newyear_syn["is_festive_evening"] = 1
        newyear_syn["pm25_spike_ratio"] = newyear_syn["pm25_spike_ratio"].clip(lower=1.5)

    # 3. Future-Year General Festive (Nov/Dec)
    general_syn = new_samples.iloc[idx2:].copy()
    if not general_syn.empty:
        general_syn["month"] = 11
        general_syn["is_diwali"] = 0
        general_syn["is_festive"] = 1
        general_syn["festive_proximity"] = 0
        general_syn["is_festive_evening"] = 1

    balanced_new = pd.concat([diwali_syn, newyear_syn, general_syn])
    balanced_new["source_label"] = minority_class
    
    df_augmented = pd.concat([df, balanced_new], ignore_index=True)
    print(f"   ✓ Added {len(balanced_new)} synthetic '{minority_class}' samples with multi-seasonal context.")
    return df_augmented


def main():
    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Delhi Air Pollution — Data Pipeline                    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Step 1
    df_pm25 = load_pm25()

    # Step 2
    pm25_min = df_pm25["datetime_hour"].min()
    pm25_max = df_pm25["datetime_hour"].max()
    df_weather = fetch_weather(pm25_min, pm25_max)

    # Step 3
    df_fire = load_fire_data()

    # Step 4
    extra = fetch_extra_pollutants()

    # Step 5
    df = merge_all(df_pm25, df_weather, df_fire, extra)

    # Step 6
    df = engineer_features(df)

    # Step 7
    df = generate_source_labels(df)

    # Step 8: Augment firecrackers
    df = augment_minority_class(df, "firecrackers", target_count=300)

    # ── Save ──
    out_path = DATA_DIR / "model_ready_delhi.csv"
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"✅ DONE! Saved {len(df)} rows × {len(df.columns)} columns")
    print(f"   File: {out_path}")
    print(f"   Time: {elapsed:.1f}s")
    print("=" * 60)

    # Quick validation
    print("\n📊 Final DataFrame Summary:")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['datetime_hour'].min()} → {df['datetime_hour'].max()}")
    print(f"   Stations: {df['location_name'].nunique()}")
    weather_pct = df["wind_speed_10m"].notna().mean() * 100
    print(f"   Weather coverage: {weather_pct:.1f}%")
    print(f"   PM2.5 NaN: {df['pm25'].isna().sum()}")
    print(f"\n   Columns:\n   {list(df.columns)}")

if __name__ == "__main__":
    main()
