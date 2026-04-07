"""
URBAN-AIRNet | train_model.py
==============================
1. Loads CPCB Chennai data (2023, 2024, 2025)
2. Cleans and preprocesses
3. Trains XGBoost + Random Forest
4. Saves models as .pkl files
5. Prints RMSE, MAE, R² for both
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "models"
FILES     = ["cpcb_2023.csv", "cpcb_2024.csv", "cpcb_2025.csv"]
TARGET    = "Ozone"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Step 1: Load and merge all years ─────────────────────────────────────────
print("=" * 55)
print("  URBAN-AIRNet — Model Training")
print("=" * 55)

dfs = []
for f in FILES:
    path = os.path.join(DATA_DIR, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"[LOAD] {f} — {len(df)} rows")
    else:
        print(f"[WARN] {f} not found, skipping.")

df = pd.concat(dfs, ignore_index=True)
print(f"\n[INFO] Total rows merged: {len(df)}")
print(f"[INFO] Columns: {list(df.columns)}")

# ── Step 2: Rename columns ────────────────────────────────────────────────────
# Standardize column names (strip spaces, lowercase)
df.columns = df.columns.str.strip().str.replace(" ", "_")
print(f"\n[INFO] Cleaned columns: {list(df.columns)}")

# ── Step 3: Select features ───────────────────────────────────────────────────
# Auto-detect available feature columns
POSSIBLE_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
    "Ozone", "WS", "WD", "RH", "RF", "TOT-RF", "SR", "BP",
    "AT", "Benzene", "Toluene", "Xylene",
    "PM2.5_(Âµg", "NO2_(Âµg", "Ozone_(Âµg"  # alternate encodings
]

# Clean column names for matching
col_map = {c: c.strip().replace(" ", "_") for c in df.columns}
df = df.rename(columns=col_map)

# Find target column
target_col = None
for c in df.columns:
    if TARGET.lower() in c.lower():
        target_col = c
        break

if target_col is None:
    # fallback: pick column with NO2 anywhere
    for c in df.columns:
        if "NO2" in c or "no2" in c.lower():
            target_col = c
            break

print(f"\n[TARGET] Using column: '{target_col}' as prediction target")

# Select numeric feature columns (exclude timestamp/date)
exclude = ["Timestamp", "Date", "Time", "Station", "City", "State"]
feature_cols = [
    c for c in df.columns
    if c != target_col
    and c not in exclude
    and df[c].dtype in [np.float64, np.int64, float, int]
]
print(f"[FEATURES] {len(feature_cols)} features: {feature_cols}")

# ── Step 4: Clean data ────────────────────────────────────────────────────────
df_clean = df[feature_cols + [target_col]].copy()

# Convert all to numeric
for c in df_clean.columns:
    df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

# Drop rows where target is missing
df_clean = df_clean.dropna(subset=[target_col])

# Fill missing features with column median
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

# Remove outliers (values beyond 3 std)
for c in [target_col]:
    mean, std = df_clean[c].mean(), df_clean[c].std()
    df_clean = df_clean[
        (df_clean[c] >= mean - 3*std) &
        (df_clean[c] <= mean + 3*std)
    ]

print(f"\n[CLEAN] Rows after cleaning: {len(df_clean)}")
print(f"[CLEAN] Target stats:\n{df_clean[target_col].describe().round(2)}")

# ── Step 5: Train / Test split ────────────────────────────────────────────────
X = df_clean[feature_cols]
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

# ── Step 6: Train XGBoost ─────────────────────────────────────────────────────
print("\n[XGB] Training XGBoost ...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae  = mean_absolute_error(y_test, xgb_pred)
xgb_r2   = r2_score(y_test, xgb_pred)

print(f"[XGB] RMSE: {xgb_rmse:.4f}")
print(f"[XGB] MAE : {xgb_mae:.4f}")
print(f"[XGB] R²  : {xgb_r2:.4f}")

# ── Step 7: Train Random Forest ───────────────────────────────────────────────
print("\n[RF] Training Random Forest ...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)

print(f"[RF]  RMSE: {rf_rmse:.4f}")
print(f"[RF]  MAE : {rf_mae:.4f}")
print(f"[RF]  R²  : {rf_r2:.4f}")

# ── Step 8: Compare models ────────────────────────────────────────────────────
print("\n── Model Comparison ─────────────────────────────────")
print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print(f"{'XGBoost':<20} {xgb_rmse:>8.4f} {xgb_mae:>8.4f} {xgb_r2:>8.4f}")
print(f"{'Random Forest':<20} {rf_rmse:>8.4f} {rf_mae:>8.4f} {rf_r2:>8.4f}")
best = "XGBoost" if xgb_r2 > rf_r2 else "Random Forest"
print(f"\n[BEST] {best} performs better (higher R²)")

# ── Step 9: Save models ───────────────────────────────────────────────────────
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_o3.pkl"))
joblib.dump(rf_model,  os.path.join(MODEL_DIR, "rf_o3.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols_o3.pkl"))
print(f"\n[SAVE] Models saved to /{MODEL_DIR}/")
print("[DONE] Training complete!")