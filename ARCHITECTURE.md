

```text
┌─────────────────────────────────┐       ┌───────────────────────────────┐
│     EXTERNAL DATA SOURCES       │       │    MODEL TRAINING PIPELINE    │
│  (Real-time & Historical)       │       │       (Off-line/Periodic)     │
└────────┬────────────────────────┘       └───────────────┬───────────────┘
         │                                                │
         ▼                                                ▼
┌───────────────────────────────┐         ┌───────────────────────────────┐
│       DATA INGESTION          │         │      HEURISTIC LABELING       │
│  - Open-Meteo (Weather)       │         │  (Creating Ground Truth from  │
│  - PM2.5 Sensors (DPCC)       │         │   Meteorological Fingerprints)│
│  - NASA FIRMS (Satellite Fire)│         └───────────────┬───────────────┘
└────────┬──────────────────────┘                         │
         │                                                ▼
         ▼                                ┌───────────────────────────────┐
┌───────────────────────────────┐         │      XGBOOST CLASSIFIER       │
│     FEATURE ENGINEERING       │────────▶│      (Training on 52+         │
│  - Festive Proximity          │         │       Spatial/Temporal Features)│
│  - Stagnation Index           │         └───────────────┬───────────────┘
│  - PM2.5 Spike Ratios         │                         │
└────────┬──────────────────────┘                         │
         │                                                │
         │             ┌──────────────────────────────────┘
         │             │
         ▼             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          BACKEND API (FastAPI)                          │
│                                                                         │
│    ┌──────────────────┐      ┌──────────────────┐      ┌─────────────┐  │
│    │ Inference Engine │◀─────┤  Source Predictor│◀─────┤ Gemini LLM  │  │
│    └────────┬─────────┘      └──────────────────┘      └─────────────┘  │
└─────────────┼───────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js)                        │
│                                                                         │
│    ┌──────────────────┐      ┌──────────────────┐      ┌─────────────┐  │
│    │ Dynamic Map      │      │ Analysis Panels  │      │ AI Insights │  │
│    │ (Leaflet/Mapbox) │      │ (Sparklines)     │      │ (Explainer) │  │
│    └──────────────────┘      └──────────────────┘      └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Functional Layers

### 1. Ingestion Layer
Reliably pulls data from heterogeneous sources including meteorological APIs and regional air quality sensor networks.

### 2. Processing & Feature Layer (`data_pipeline.py`)
Transforms raw metrics into "Pollution Fingerprints." This is where we calculate indices like **Stagnation** (trapping efficiency) and **Festive Proximity** (Diwali/New Year context).

### 3. Model Layer (`source_model.py`)
An XGBoost multi-class classifier that has mapped 52+ variables to 10 specific pollution sources. Handles the weighted inference for real-time predictions.

### 4. Application Layer (`main.py`)
The bridge between ML and the User.
- **FastAPI**: Manages real-time data merging for the user's specific location/time.
- **LLM Explainer**: Uses Google Gemini to translate abstract model outputs into natural language insights for policy makers or the public.

### 5. Presentation Layer (`frontend/`)
A high-performance React dashboard designed for "Situation Room" visibility, featuring geospatial visualization and temporal "Time Travel" navigation.
