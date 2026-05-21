# 🌫️ URBAN-AIRNet

> **AI Framework for Urban Air Pollution Forecasting**
> Predicts ozone and pollutant levels across high-traffic road intersections in Chennai using CPCB sensor data, graph-based road analytics, and ensemble machine learning.

---

## 📌 Overview

URBAN-AIRNet combines **real-world CPCB air quality data** with **OpenStreetMap road network analysis** to forecast pollution at urban intersections. It trains dual ML models (XGBoost + Random Forest), selects the best performer, and surfaces predictions through an interactive dashboard.

**City focus:** Chennai, Tamil Nadu, India
**Data source:** CPCB (Central Pollution Control Board) — 2023, 2024, 2025
**Target pollutant:** Ozone (O₃), with full multi-pollutant feature support

---

## 🗂️ Project Structure

```
URBAN-AIRNet/
├── data/                        # CPCB CSV datasets (cpcb_2023.csv, cpcb_2024.csv, cpcb_2025.csv)
├── models/                      # Saved model artifacts (.pkl)
│   ├── xgboost_o3.pkl
│   ├── rf_o3.pkl
│   └── feature_cols_o3.pkl
├── dashboard/                   # Interactive visualization app
│   └── data/
│       └── intersections.csv    # Preprocessed Chennai intersection data
├── cache/                       # Cached intermediate outputs
├── .devcontainer/               # VS Code dev container config
├── generate_intersections.py    # OSM road graph → intersection CSV
├── train_model.py               # Model training pipeline
└── README.md
```

---

## ⚙️ How It Works

### 1. Intersection Extraction (`generate_intersections.py`)
- Downloads Chennai's road network from **OpenStreetMap** via `osmnx`
- Computes **betweenness centrality** using `networkx` to rank intersection traffic load
- Filters to the **top 25% busiest intersections** (degree ≥ 3, road score threshold)
- Scores intersections by dominant road type (motorway → residential)
- Scales pollutant estimates (NOx, CO, PM2.5) by road type score
- Outputs `dashboard/data/intersections.csv`

### 2. Model Training (`train_model.py`)
- Loads and merges 3 years of CPCB data
- Auto-detects available feature columns; standardizes naming
- Cleans data: numeric coercion, median imputation, 3σ outlier removal
- Trains two regressors on an 80/20 split:

| Model | Config |
|---|---|
| **XGBoost** | 200 trees, depth 6, lr 0.05, subsample 0.8 |
| **Random Forest** | 200 trees, depth 10, min_samples_split 5 |

- Evaluates both on RMSE, MAE, R² — reports the winner
- Saves all artifacts to `/models/`

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost joblib osmnx networkx
```

### Step 1 — Generate intersection data

```bash
python generate_intersections.py
```

> Downloads Chennai road graph from OSM (~1–3 min depending on connection).
> Output: `dashboard/data/intersections.csv`

### Step 2 — Train the models

```bash
python train_model.py
```

> Expects CPCB CSVs in `/data/`. Outputs `.pkl` model files to `/models/`.

### Step 3 — Launch the dashboard

```bash
cd dashboard
# Follow dashboard-specific instructions (Streamlit/Dash/etc.)
```

---

## 📊 Input Data Format

Place CPCB CSV files in `/data/` named:
- `cpcb_2023.csv`
- `cpcb_2024.csv`
- `cpcb_2025.csv`

Expected columns (auto-detected, flexible naming):

| Feature | Description |
|---|---|
| `PM2.5`, `PM10` | Particulate matter (µg/m³) |
| `NO`, `NO2`, `NOx` | Nitrogen oxides (ppb / µg/m³) |
| `CO`, `SO2`, `NH3` | Gaseous pollutants |
| `Ozone` | **Target variable** |
| `AT`, `RH`, `WS`, `WD` | Meteorological features |
| `SR`, `BP`, `RF` | Solar radiation, pressure, rainfall |

---

## 📈 Model Output

After training, the console reports a comparison table:

```
── Model Comparison ─────────────────────────────────
Model                    RMSE      MAE       R²
XGBoost              x.xxxx    x.xxxx    x.xxxx
Random Forest        x.xxxx    x.xxxx    x.xxxx

[BEST] XGBoost performs better (higher R²)
```

Saved artifacts:
- `models/xgboost_o3.pkl` — trained XGBoost regressor
- `models/rf_o3.pkl` — trained Random Forest regressor
- `models/feature_cols_o3.pkl` — ordered feature list for inference

---

## 🗺️ Dashboard

The dashboard uses the trained models and intersection data to visualize predicted pollution levels across Chennai's road network on an interactive map. Real-time meteorological values from CPCB override the default climatological means at inference time.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data | pandas, numpy |
| Geospatial | osmnx, networkx |
| ML | scikit-learn, xgboost |
| Persistence | joblib |
| Visualization | dashboard/ (Streamlit / Plotly) |

---

## 📋 Roadmap

- [ ] LSTM-based temporal forecasting
- [ ] Multi-city support (beyond Chennai)
- [ ] Real-time CPCB API integration
- [ ] AQI category prediction (alongside continuous regression)
- [ ] Docker container for one-command deployment

---

## 📄 License

This project is open-source. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [CPCB India](https://cpcb.nic.in/) for air quality monitoring data
- [OpenStreetMap](https://www.openstreetmap.org/) contributors for road network data
- [osmnx](https://github.com/gboeing/osmnx) by Geoff Boeing
