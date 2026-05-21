# URBAN-AIRNet
AI Framework for Urban Air Pollution Forecasting

-------------------------------------------------------------

Install dependencies
bashpip install pandas numpy scikit-learn xgboost joblib osmnx networkx
Step 1 — Generate intersection data
bashpython generate_intersections.py

Downloads Chennai road graph from OSM (~1–3 min depending on connection).
Output: dashboard/data/intersections.csv

Step 2 — Train the models
bashpython train_model.py

Expects CPCB CSVs in /data/. Outputs .pkl model files to /models/.

Step 3 — Launch the dashboard
bashcd dashboard
# Follow dashboard-specific instructions (Streamlit/Dash/etc.)

📊 Input Data Format
Place CPCB CSV files in /data/ named:

cpcb_2023.csv
cpcb_2024.csv
cpcb_2025.csv

Expected columns (auto-detected, flexible naming):
FeatureDescriptionPM2.5, PM10Particulate matter (µg/m³)NO, NO2, NOxNitrogen oxides (ppb / µg/m³)CO, SO2, NH3Gaseous pollutantsOzoneTarget variableAT, RH, WS, WDMeteorological featuresSR, BP, RFSolar radiation, pressure, rainfall

📈 Model Output
After training, the console reports a comparison table:
── Model Comparison ─────────────────────────────────
Model                    RMSE      MAE       R²
XGBoost              x.xxxx    x.xxxx    x.xxxx
Random Forest        x.xxxx    x.xxxx    x.xxxx

[BEST] XGBoost performs better (higher R²)
Saved artifacts:

models/xgboost_o3.pkl — trained XGBoost regressor
models/rf_o3.pkl — trained Random Forest regressor
models/feature_cols_o3.pkl — ordered feature list for inference


🗺️ Dashboard
The dashboard uses the trained models and intersection data to visualize predicted pollution levels across Chennai's road network on an interactive map. Real-time meteorological values from CPCB override the default climatological means at inference time.

🛠️ Tech Stack
LayerToolsDatapandas, numpyGeospatialosmnx, networkxMLscikit-learn, xgboostPersistencejoblibVisualizationdashboard/ (Streamlit / Plotly)

📋 Roadmap

 LSTM-based temporal forecasting
 Multi-city support (beyond Chennai)
 Real-time CPCB API integration
 AQI category prediction (alongside continuous regression)
 Docker container for one-command deployment


📄 License
This project is open-source. See LICENSE for details.

🙏 Acknowledgements

CPCB India for air quality monitoring data
OpenStreetMap contributors for road network data
osmnx by Geoff Boeing
