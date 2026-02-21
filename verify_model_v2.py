import pandas as pd
import xgboost as xgb
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Load model and metadata
MODEL_PATH = "models/source_classifier.json"
META_PATH = "models/model_metadata.json"

with open(META_PATH) as f:
    meta = json.load(f)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

CLASS_NAMES = meta["class_names"]
FEATURE_COLS = meta["feature_columns"]

def test_scenario(name, pm25, hum, wind, timestamp, spike_ratio=1.0):
    # Mock feature building
    dt = pd.to_datetime(timestamp)
    # Mirroring main.py logic
    diwali_2025 = {"2025-10-18","2025-10-19","2025-10-20","2025-10-21","2025-10-22"}
    festive_dates = {
        "2024-10-30", "2024-10-31", "2024-11-01", "2024-11-02", "2024-11-03",
        "2025-10-18", "2025-10-19", "2025-10-20", "2025-10-21", "2025-10-22",
        "2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10",
        "2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01"
    }
    
    date_str = dt.strftime("%Y-%m-%d")
    is_diwali = 1 if date_str in diwali_2025 else 0
    is_festive = 1 if date_str in festive_dates else 0
    
    # Proximity (simplified)
    proximity = 0 if is_festive else 15
    hour = dt.hour
    is_festive_evening = 1 if (proximity <= 2 and (hour >= 16 or hour <= 4) and pm25 > 120) else 0
    
    features = {c: 0.0 for c in FEATURE_COLS}
    features.update({
        "pm25": pm25,
        "temperature_2m": 20,
        "relative_humidity_2m": hum,
        "wind_speed_10m": wind,
        "is_diwali": is_diwali,
        "is_festive": is_festive,
        "festive_proximity": proximity,
        "is_festive_evening": is_festive_evening,
        "pm25_spike_ratio": spike_ratio,
        "month": dt.month,
        "hour_ist": hour,
        "wind_speed_inv": 1.0 / (wind + 0.1),
        "stagnation_index": (1.0 / (wind + 0.1)) * hum
    })
    
    X = pd.DataFrame([features])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    
    print(f"\nScenario: {name}")
    print(f"  Input: PM2.5={pm25}, Wind={wind}, Date={timestamp}")
    print(f"  Result: {CLASS_NAMES[pred_idx]} ({proba[pred_idx]:.4f})")
    
    # Print top 3 probabilities
    top3_idx = np.argsort(proba)[-3:][::-1]
    print(f"  Top probabilities: {[f'{CLASS_NAMES[i]}: {proba[i]:.4f}' for i in top3_idx]}")

print("Verifying Retrained Model...")

# 1. Diwali 2025 Peak (Known date)
test_scenario("Diwali 2025 Peak", 450, 65, 1.2, "2025-10-20 21:00:00", spike_ratio=3.5)

# 2. Diwali 2024 (Now newly supported)
test_scenario("Diwali 2024 Peak", 380, 60, 1.5, "2024-11-01 22:00:00", spike_ratio=2.8)

# 3. New Year (Newly supported)
test_scenario("New Year midnight", 280, 75, 1.0, "2026-01-01 00:30:00", spike_ratio=2.0)

# 4. Biomass scenario (NW wind, Oct, no festive peak)
test_scenario("Typical Biomass (Oct, NW wind)", 200, 45, 4.5, "2025-10-10 14:00:00", spike_ratio=1.1)

# 5. Winter Inversion (Dec night, very high PM2.5, no festive)
test_scenario("Winter Inversion (Dec Night)", 350, 85, 0.8, "2025-12-15 02:00:00", spike_ratio=1.2)
