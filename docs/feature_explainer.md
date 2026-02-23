# PIE: Feature Engineering & Classification Logic

This document provides a detailed breakdown of the features used by the Pollution Intelligence Engine (PIE) and how they influence the source attribution model.

---

## 1. Feature Origin: Fetched vs. Engineered

The model uses a total of **52+ features**. These are split between raw data pulled from sensors/APIs and complex indicators calculated by our pipeline.

### A. Directly Fetched Features
These are "raw" values retrieved from environmental monitoring stations and global APIs:
*   **Pollutants**: `pm25` (µg/m³), `no2` (µg/m³), `pm10` (µg/m³).
*   **Weather (Open-Meteo)**: `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `surface_pressure`, `precipitation`, `cloud_cover`.
*   **Satellite Data (NASA FIRMS)**: Active fire coordinates (Latitude/Longitude) and brightness.
*   **Spatial Constants**: Station `latitude` and `longitude`.

### B. Engineered Features
These are the "Pollution Fingerprints" we calculate to provide context:
*   **Temporal Context**:
    *   `hour_ist`: The 24-hour cycle in local time.
    *   `is_rush_hour`: Flags 8-10 AM and 5-8 PM (Traffic peaks).
    *   `is_night`: Captures night-time cooling and the "mixing layer" collapse.
*   **Meteorological Indices**:
    *   `wind_speed_inv`: The inverse of wind intensity to amplify local signals.
        *   **Formula**: `1.0 / (wind_speed_10m + 0.1)`
    *   `stagnation_index`: Measures the combined "trapping" effect of calm wind and moisture.
        *   **Formula**: `wind_speed_inv * relative_humidity_2m`
    *   `pm25_spike_ratio`: Compares current PM2.5 to the 12-hour average to detect "burst" events.
        *   **Formula**: `pm25 / (pm25_rolling_12h + 0.1)`
*   **Spatial Indicators**:
    *   `dist_to_highways`: Minimum distance to pre-mapped arterial roads.
        *   **Formula**: `min(Haversine(coord, highway_nodes))`
    *   `upwind_fire_flag`: Uses wind direction to validate satellite observations.
        *   **Logic**: `satellite_fire_detected` AND `wind_direction` points from fire to station.
*   **Festive Awareness (NEW)**:
    *   `festive_proximity`: Days until/since the nearest major festival.
        *   **Formula**: `min(abs(date - festival_dates))` clipped at 15.
    *   `is_festive_evening`: A high-confidence fingerprint for festive events.
        *   **Logic**: `(festive_proximity <= 2)` AND `(hour in [16:00 - 03:00])` AND `(pm25 > 120)`

---

## 2. Impact on Classification Categories

How the model uses these features to distinguish between sources:

| Source Category | Key Driving Features | Logic / Influence |
| :--- | :--- | :--- |
| **Traffic** | `no2`, `is_rush_hour`, `dist_to_highways` | High NO2 combustion markers + proximity to roads during peak commute times. |
| **Biomass** | `upwind_fire_flag`, `month`, `wind_direction` | Active fires upwind + North-Westerly winds during the October/November harvesting season. |
| **Firecrackers** | `is_festive_evening`, `pm25_spike_ratio` | Sharp, intense PM2.5 spikes (2-3x baseline) occurring specifically during festive nights. |
| **Winter Inversion** | `cold_night_flag`, `is_night`, `month` | Occurs during Dec–Feb when cold air traps local heating and cooking smoke near the ground. |
| **Stagnation** | `stagnation_index`, `wind_speed_10m` | General accumulation of pollution due to 0-3 km/h wind speeds and lack of atmospheric mixing. |
| **Secondary Aerosols** | `relative_humidity_2m`, `stagnation_index` | High humidity (>80%) combined with calm winds allows chemical reactions to form new particles in the air. |
| **Soil & Road Dust** | `pm10/pm25 ratio`, `wind_speed_10m` | A high concentration of PM10 (coarse particles) relative to PM2.5, often exacerbated by high wind gusting. |
| **Clean Air** | `pm25 < 30` | Identified primarily by absolute pollutant thresholds being below safe limits. |
| **Background** | `low wind_speed_inv`, `stable pm25` | Low-level, stable regional pollution with no local spikes detected. |

---

## 3. Summary of Influence
*   **Weather** acts as a "Filter": It determines whether pollution *can* accumulate.
*   **Spacial/Satellite** acts as a "Pointer": It identifies specific source locations.
*   **Temporal/Festive** acts as a "Context": It maps the pollution to human behavior.
