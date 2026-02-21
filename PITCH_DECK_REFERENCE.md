# PIE: Pollution Intelligence Engine - Technical Pitch Reference

## 1. The Problem: The "Blame Game" of Air Pollution
Delhi's air quality crisis is complex. Policy makers often struggle with source attribution: Is today's smog from crop burning (biomass), vehicle exhaust, or road dust? Currently, most monitoring only tells us **how much** pollution exists, not **where** it comes from in real-time.

## 2. Our Solution: REAL-TIME Source Attribution
PIE is a Machine Learning-based system that classifies the primary source of PM2.5 pollution for any given reading. It uses an **XGBoost Multi-class Classifier** trained on a combination of meteorological, spatial, and temporal data to provide an immediate "source fingerprint."

---

## 3. Feature Engineering (The "Secret Sauce")
We transform raw sensor data into meaningful signals. We use **52 distinct features** categorized into five layers:

### A. Meteorological Signal (Dispersion & Trapping)
- **Stagnation Index**: Combines low wind speed and high humidity to detect when weather "traps" pollution (inversion).
- **Dispersion Index**: Measures how effectively the atmosphere is "cleaning" itself.
- **Wind Speed Inverse**: Amplifies the signal of local vs. long-range transport.

### B. Temporal Signal (Human Activity Patterns)
- **Rush Hour Flags**: Identifies morning/evening peaks characteristic of traffic.
- **Weekend/Night Flags**: Captures industrial activity shifts and evening cooling effects.

### C. NEW: Festive Context Signal (The Diwali Fix)
*Recently engineered to solve "Firecracker Detection" failures:*
- **Festive Proximity**: A continuous gradient (0-15 days) to the nearest major festival.
- **is_festive_evening**: A binary fingerprint capturing the intersection of festive proximity, night-time hours, and high pollution.
- **PM2.5 Spike Ratio**: Compares current reading to a 12-hour baseline. Firecrackers create "sharp" 2-3x spikes, whereas biomass/traffic increases are more gradual.

### D. Spatial Signal (Proximity to Sources)
- **Distance to Landmarks**: Precise distance to major highways (Traffic), industrial zones, and brick kilns.
- **Upwind Fire Flag**: Uses real-time satellite data to check if active fires exist in the direction the wind is blowing from.

### E. Chemical Signatures
- **PM2.5 to NO2 Ratio**: High NO2 usually signals vehicular combustion (Traffic).
- **PM10/PM2.5 Ratio**: A high ratio suggests "Coarse" pollution like road dust or construction.

---

## 4. The 10 Classification Classes
The model categorizes pollution into 10 distinct sources:

1.  **Urban Mix**: General city pollution (baseline).
2.  **Traffic**: High localized vehicular emissions (Rush hour + NO2).
3.  **Biomass**: Crop/Stubble burning upwind (October-November peak).
4.  **Firecrackers**: Intense episodic spikes during festivals (Diwali/New Year).
5.  **Soil & Road Dust**: Coarse particulate matter from construction/arid ground.
6.  **Secondary Aerosols**: Chemical reactions in-air (High humidity + secondary markers).
7.  **Winter Inversion**: Cold, calm nights where local smoke is trapped near the ground.
8.  **Stagnation**: General lack of dispersion.
9.  **Background**: Lowest detectable regional pollution.
10. **Clean Air**: Readings below safe threshold.

---

## 5. How We Classify It (The Logic Flow)
The system uses a two-stage approach:

### Stage 1: Rule-Based Heuristic (The Teacher)
Since "labeled" ground truth is rare for air pollution, we use a sophisticated **Heuristic Labeler**. It looks for specific "fingerprints":
- *Example*: If it's a festive evening, PM2.5 jumped 1.5x, and wind is low → Label as **Firecrackers**.
- *Example*: If wind is from the NW in November and satellite fires are spotted → Label as **Biomass**.

### Stage 2: XGBoost Machine Learning (The Expert)
The XGBoost model learns these complex patterns and generalizes them. Even if a specific rule is borderline, the model looks at all 52 features simultaneously to make a prediction with a confidence score.

---

## 6. Competitive Advantages
- **No Extra Hardware**: Uses existing sensor networks + weather APIs.
- **Multi-Year Awareness**: Supports festive detection across 2024-2026.
- **Explainability**: Integrated with SHAP analysis to show *why* a source was predicted (e.g., "Predicted Traffic because NO2 is high and it's 6 PM").
