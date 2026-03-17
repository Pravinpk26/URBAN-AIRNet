"""
URBAN-AIRNet — Streamlit Dashboard
====================================
Run with: streamlit run app.py
Install : pip install streamlit folium streamlit-folium plotly pandas numpy shap scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="URBAN-AIRNet | Chennai",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark industrial theme */
.stApp {
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1e2433;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #151922 0%, #1a2035 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 6px;
}
.metric-delta {
    font-size: 0.8rem;
    margin-top: 4px;
}

/* Section headers */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 4px;
}
.section-heading {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 20px;
}

/* AQI badge */
.aqi-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2d4a, transparent);
    margin: 32px 0;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #111318;
    border-bottom: 1px solid #1e2433;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: #64748b;
    background: transparent;
    border: none;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)


# ── Dummy Data Generators ─────────────────────────────────────────────────────

@st.cache_data
def generate_grid_data():
    """Generate dummy pollution + road feature grid for Chennai."""
    np.random.seed(42)
    lats = np.arange(13.02, 13.33, 0.01)
    lons = np.arange(80.12, 80.33, 0.01)
    records = []
    for lat in lats:
        for lon in lons:
            # Simulate higher pollution near city center and major roads
            dist_center = np.sqrt((lat - 13.08)**2 + (lon - 80.27)**2)
            base_no2 = 45 - dist_center * 80 + np.random.normal(0, 5)
            base_o3  = 30 + dist_center * 20 + np.random.normal(0, 4)
            records.append({
                "lat": round(lat, 3),
                "lon": round(lon, 3),
                "NO2_ppb":              max(5,  round(base_no2, 2)),
                "O3_ppb":               max(10, round(base_o3,  2)),
                "node_degree_mean":     round(np.random.uniform(2, 5), 2),
                "intersection_density": round(np.random.uniform(0.5, 8), 2),
                "betweenness_centrality": round(np.random.uniform(0, 0.01), 5),
                "road_type_score":      round(np.random.uniform(0.1, 0.9), 2),
            })
    return pd.DataFrame(records)


@st.cache_data
def generate_forecast_data():
    """Generate 48-hour forecast time series."""
    np.random.seed(7)
    now   = datetime.now().replace(minute=0, second=0, microsecond=0)
    hours = [now + timedelta(hours=i) for i in range(49)]

    no2_actual = [38 + 10 * np.sin(i * 0.3) + np.random.normal(0, 3) for i in range(25)]
    no2_pred   = no2_actual[-1:] + [
        no2_actual[-1] + 8 * np.sin((i + 24) * 0.3) + np.random.normal(0, 2)
        for i in range(1, 25)
    ]
    o3_actual  = [28 + 8 * np.sin(i * 0.25 + 1) + np.random.normal(0, 2) for i in range(25)]
    o3_pred    = o3_actual[-1:] + [
        o3_actual[-1] + 6 * np.sin((i + 24) * 0.25 + 1) + np.random.normal(0, 2)
        for i in range(1, 25)
    ]
    return {
        "hours":      hours,
        "no2_actual": no2_actual,
        "no2_pred":   no2_pred,
        "o3_actual":  o3_actual,
        "o3_pred":    o3_pred,
    }


@st.cache_data
def generate_shap_data():
    """Generate dummy SHAP feature importance values."""
    features = [
        "road_type_score",
        "intersection_density",
        "wind_speed_ms",
        "betweenness_centrality",
        "node_degree_mean",
        "temperature_C",
        "relative_humidity",
        "boundary_layer_height",
        "NO2_lag_1d",
        "solar_radiation",
    ]
    shap_no2 = [0.312, 0.278, 0.198, 0.165, 0.142, 0.118, 0.095, 0.082, 0.071, 0.045]
    shap_o3  = [0.145, 0.132, 0.289, 0.098, 0.087, 0.241, 0.187, 0.156, 0.062, 0.134]
    return pd.DataFrame({
        "feature": features,
        "SHAP_NO2": shap_no2,
        "SHAP_O3":  shap_o3,
    })


def aqi_color(value, pollutant="NO2"):
    """Return color based on AQI level."""
    if pollutant == "NO2":
        if value < 20:  return "#22c55e"
        if value < 35:  return "#84cc16"
        if value < 50:  return "#eab308"
        if value < 65:  return "#f97316"
        return "#ef4444"
    else:
        if value < 20:  return "#22c55e"
        if value < 30:  return "#84cc16"
        if value < 45:  return "#eab308"
        if value < 60:  return "#f97316"
        return "#ef4444"


def aqi_label(value):
    if value < 20:  return "Good",     "#22c55e"
    if value < 35:  return "Moderate", "#84cc16"
    if value < 50:  return "Unhealthy","#eab308"
    if value < 65:  return "Poor",     "#f97316"
    return "Hazardous", "#ef4444"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem; 
                    font-weight: 700; color: #3b82f6; letter-spacing: 0.05em;'>
            URBAN-AIRNet
        </div>
        <div style='font-size: 0.75rem; color: #475569; margin-top: 2px;'>
            Road-Aware Air Pollution Forecasting
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**City**")
    st.selectbox("", ["Chennai, Tamil Nadu"], label_visibility="collapsed")

    st.markdown("**Pollutant**")
    pollutant = st.radio("", ["NO₂", "O₃", "Both"], label_visibility="collapsed")

    st.markdown("**Forecast horizon**")
    horizon = st.slider("", 6, 48, 24, 6, label_visibility="collapsed", format="%dh")

    st.markdown("**Date range**")
    col1, col2 = st.columns(2)
    with col1:
        st.date_input("From", datetime(2022, 1, 1), label_visibility="collapsed")
    with col2:
        st.date_input("To",   datetime(2022, 3, 31), label_visibility="collapsed")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 0.7rem; color: #334155; line-height: 1.8;'>
        <b style='color:#475569'>Data sources</b><br>
        · TROPOMI Sentinel-5P<br>
        · ERA5 / MERRA-2<br>
        · OpenStreetMap (OSMnx)<br><br>
        <b style='color:#475569'>Models</b><br>
        · XGBoost · Random Forest<br>
        · LSTM Seq2Seq<br><br>
        <b style='color:#475569'>XAI</b><br>
        · SHAP Feature Importance
    </div>
    """, unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
df_grid     = generate_grid_data()
forecast    = generate_forecast_data()
df_shap     = generate_shap_data()

avg_no2 = df_grid["NO2_ppb"].mean()
avg_o3  = df_grid["O3_ppb"].mean()
max_no2 = df_grid["NO2_ppb"].max()
aqi_lbl, aqi_clr = aqi_label(avg_no2)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; justify-content:space-between; 
            margin-bottom: 24px; padding-bottom: 16px; 
            border-bottom: 1px solid #1e2433;'>
    <div>
        <div style='font-family: Space Mono, monospace; font-size: 0.65rem; 
                    letter-spacing: 0.2em; color: #3b82f6; text-transform: uppercase;'>
            Urban Air Quality Intelligence
        </div>
        <div style='font-size: 1.6rem; font-weight: 600; color: #f1f5f9; 
                    margin-top: 2px;'>
            Chennai Air Pollution Dashboard
        </div>
    </div>
    <div style='text-align:right; font-size:0.75rem; color:#475569; 
                font-family: Space Mono, monospace;'>
        LIVE SIMULATION<br>
        <span style='color:#22c55e'>● </span>Model Active
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#60a5fa'>{avg_no2:.1f} <span style='font-size:1rem'>ppb</span></div>
        <div class='metric-label'>Mean NO₂ · Chennai</div>
        <div class='metric-delta' style='color:#f97316'>▲ 12% vs yesterday</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#a78bfa'>{avg_o3:.1f} <span style='font-size:1rem'>ppb</span></div>
        <div class='metric-label'>Mean O₃ · Chennai</div>
        <div class='metric-delta' style='color:#22c55e'>▼ 4% vs yesterday</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:{aqi_clr}'>{aqi_lbl}</div>
        <div class='metric-label'>Current AQI Status</div>
        <div class='metric-delta' style='color:#64748b'>Based on NO₂ + O₃</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#34d399'>{max_no2:.1f} <span style='font-size:1rem'>ppb</span></div>
        <div class='metric-label'>Peak NO₂ · Hotspot</div>
        <div class='metric-delta' style='color:#64748b'>Anna Salai corridor</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺  AQI Heatmap",
    "🛣  Road Network",
    "📈  Forecast",
    "🔍  SHAP Analysis",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — AQI Heatmap
# ════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown("<div class='section-title'>Spatial Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>NO₂ / O₃ Concentration Map</div>", unsafe_allow_html=True)

        poll_choice = st.radio(
            "Show pollutant:",
            ["NO₂", "O₃"],
            horizontal=True,
            key="map_poll"
        )
        poll_col = "NO2_ppb" if poll_choice == "NO₂" else "O3_ppb"

        # Build Folium map
        m = folium.Map(
            location=[13.08, 80.27],
            zoom_start=12,
            tiles="CartoDB dark_matter",
        )

        # Add heatmap circles
        for _, row in df_grid.iterrows():
            val   = row[poll_col]
            color = aqi_color(val, "NO2" if poll_choice == "NO₂" else "O3")
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                weight=0,
                popup=folium.Popup(
                    f"<b>{poll_choice}:</b> {val:.1f} ppb<br>"
                    f"<b>Road score:</b> {row['road_type_score']:.2f}<br>"
                    f"<b>Intersection density:</b> {row['intersection_density']:.1f}",
                    max_width=200
                ),
            ).add_to(m)

        st_folium(m, width=None, height=480, returned_objects=[])

    with col_right:
        st.markdown("<div class='section-title'>AQI Legend</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for label, color, rng in [
            ("Good",      "#22c55e", "< 20 ppb"),
            ("Moderate",  "#84cc16", "20–35 ppb"),
            ("Unhealthy", "#eab308", "35–50 ppb"),
            ("Poor",      "#f97316", "50–65 ppb"),
            ("Hazardous", "#ef4444", "> 65 ppb"),
        ]:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:10px; 
                        margin-bottom:12px; padding: 10px 14px;
                        background:#111318; border-radius:8px;
                        border-left: 3px solid {color}'>
                <div style='font-size:0.85rem; font-weight:500; color:{color}'>{label}</div>
                <div style='font-size:0.75rem; color:#475569; margin-left:auto'>{rng}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#151922; border:1px solid #1e2d4a; 
                    border-radius:10px; padding:16px;'>
            <div style='font-size:0.7rem; color:#3b82f6; letter-spacing:0.1em; 
                        text-transform:uppercase; margin-bottom:8px;'>
                Grid stats
            </div>
            <div style='font-size:0.8rem; color:#94a3b8; line-height:2'>
                Cells: {len(df_grid)}<br>
                Resolution: 0.01°<br>
                ≈ 1 km × 1 km<br>
                Coverage: Chennai
            </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — Road Network
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Infrastructure Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Road Network Feature Map</div>", unsafe_allow_html=True)

    feat_choice = st.selectbox(
        "Road feature to visualise:",
        ["road_type_score", "intersection_density", "betweenness_centrality", "node_degree_mean"],
        format_func=lambda x: {
            "road_type_score":        "Road Type Score (motorisation index)",
            "intersection_density":   "Intersection Density (nodes/km²)",
            "betweenness_centrality": "Betweenness Centrality",
            "node_degree_mean":       "Mean Node Degree",
        }[x]
    )

    col_map, col_stats = st.columns([3, 1])

    with col_map:
        m2 = folium.Map(
            location=[13.08, 80.27],
            zoom_start=12,
            tiles="CartoDB dark_matter",
        )

        feat_vals = df_grid[feat_choice]
        vmin, vmax = feat_vals.min(), feat_vals.max()

        def road_color(val):
            norm = (val - vmin) / (vmax - vmin + 1e-9)
            r = int(255 * norm)
            b = int(255 * (1 - norm))
            return f"#{r:02x}44{b:02x}"

        for _, row in df_grid.iterrows():
            val = row[feat_choice]
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color=road_color(val),
                fill=True,
                fill_color=road_color(val),
                fill_opacity=0.7,
                weight=0,
                popup=f"{feat_choice}: {val:.4f}",
            ).add_to(m2)

        st_folium(m2, width=None, height=460, returned_objects=[])

    with col_stats:
        st.markdown("<div class='section-title'>Feature Stats</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        for feat, label in [
            ("road_type_score",        "Road type score"),
            ("intersection_density",   "Intersection density"),
            ("betweenness_centrality", "Betweenness"),
            ("node_degree_mean",       "Node degree"),
        ]:
            mean_val = df_grid[feat].mean()
            st.markdown(f"""
            <div style='background:#111318; border-radius:8px; padding:12px 14px;
                        margin-bottom:10px; border:1px solid #1e2433;'>
                <div style='font-size:0.7rem; color:#64748b; 
                            text-transform:uppercase; letter-spacing:0.08em;'>
                    {label}
                </div>
                <div style='font-family: Space Mono, monospace; font-size:1.1rem; 
                            color:#60a5fa; margin-top:4px;'>
                    {mean_val:.3f}
                </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 — Forecast Chart
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Temporal Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>24–48 Hour Pollution Forecast</div>", unsafe_allow_html=True)

    hours      = forecast["hours"]
    split_idx  = 24

    actual_hours = hours[:split_idx + 1]
    pred_hours   = hours[split_idx:]

    fig = go.Figure()

    # NO2 actual
    fig.add_trace(go.Scatter(
        x=actual_hours, y=forecast["no2_actual"],
        name="NO₂ Observed", mode="lines",
        line=dict(color="#60a5fa", width=2),
    ))
    # NO2 forecast
    fig.add_trace(go.Scatter(
        x=pred_hours, y=forecast["no2_pred"],
        name="NO₂ Forecast", mode="lines",
        line=dict(color="#60a5fa", width=2, dash="dash"),
    ))
    # O3 actual
    fig.add_trace(go.Scatter(
        x=actual_hours, y=forecast["o3_actual"],
        name="O₃ Observed", mode="lines",
        line=dict(color="#a78bfa", width=2),
    ))
    # O3 forecast
    fig.add_trace(go.Scatter(
        x=pred_hours, y=forecast["o3_pred"],
        name="O₃ Forecast", mode="lines",
        line=dict(color="#a78bfa", width=2, dash="dash"),
    ))

    # Vertical line at forecast start
    fig.add_vline(
        x=hours[split_idx],
        line_width=1,
        line_dash="dot",
        line_color="#334155",
        annotation_text="Forecast →",
        annotation_font_color="#475569",
        annotation_font_size=11,
    )

    fig.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#111318",
        font=dict(family="DM Sans", color="#94a3b8"),
        legend=dict(
            bgcolor="#111318",
            bordercolor="#1e2433",
            borderwidth=1,
            font=dict(size=12),
        ),
        xaxis=dict(
            gridcolor="#1e2433",
            tickfont=dict(size=11),
            title="Time",
        ),
        yaxis=dict(
            gridcolor="#1e2433",
            tickfont=dict(size=11),
            title="Concentration (ppb)",
        ),
        hovermode="x unified",
        height=420,
        margin=dict(l=40, r=20, t=20, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Model metrics row
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, color in [
        (m1, "RMSE (NO₂)",  "3.42 ppb",  "#60a5fa"),
        (m2, "MAE (NO₂)",   "2.81 ppb",  "#60a5fa"),
        (m3, "R² Score",    "0.873",     "#34d399"),
        (m4, "Model",       "LSTM",      "#a78bfa"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#111318; border:1px solid #1e2433; border-radius:10px;
                        padding:16px; text-align:center;'>
                <div style='font-family: Space Mono, monospace; font-size:1.3rem; 
                            color:{color}; font-weight:700;'>{val}</div>
                <div style='font-size:0.72rem; color:#475569; text-transform:uppercase;
                            letter-spacing:0.1em; margin-top:4px;'>{label}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP Analysis
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>Explainable AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>SHAP Feature Importance</div>", unsafe_allow_html=True)

    shap_target = st.radio(
        "Target pollutant:",
        ["NO₂", "O₃"],
        horizontal=True,
        key="shap_poll"
    )
    shap_col = "SHAP_NO2" if shap_target == "NO₂" else "SHAP_O3"
    df_sorted = df_shap.sort_values(shap_col, ascending=True)

    col_bar, col_insight = st.columns([2, 1])

    with col_bar:
        colors = [
            "#3b82f6" if "road" in f or "intersection" in f or "betweenness" in f or "degree" in f
            else "#475569"
            for f in df_sorted["feature"]
        ]

        fig2 = go.Figure(go.Bar(
            x=df_sorted[shap_col],
            y=df_sorted["feature"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" for v in df_sorted[shap_col]],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11, family="Space Mono"),
        ))

        fig2.update_layout(
            paper_bgcolor="#0d0f14",
            plot_bgcolor="#111318",
            font=dict(family="DM Sans", color="#94a3b8"),
            xaxis=dict(
                gridcolor="#1e2433",
                title="Mean |SHAP value|",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                gridcolor="transparent",
                tickfont=dict(size=12),
            ),
            height=400,
            margin=dict(l=20, r=60, t=10, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div style='display:flex; gap:20px; margin-top:8px;'>
            <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:12px; height:12px; background:#3b82f6; border-radius:2px'></div>
                <span style='font-size:0.78rem; color:#64748b;'>Road network features</span>
            </div>
            <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:12px; height:12px; background:#475569; border-radius:2px'></div>
                <span style='font-size:0.78rem; color:#64748b;'>Meteorological features</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_insight:
        st.markdown("<div class='section-title'>Key Insights</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        road_shap = df_shap[df_shap["feature"].isin([
            "road_type_score", "intersection_density",
            "betweenness_centrality", "node_degree_mean"
        ])][shap_col].sum()
        total_shap = df_shap[shap_col].sum()
        road_pct = road_shap / total_shap * 100

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e3a5f, #1a2d4a);
                    border:1px solid #1e4d8c; border-radius:12px; padding:20px;
                    margin-bottom:16px;'>
            <div style='font-family: Space Mono, monospace; font-size:1.8rem; 
                        color:#60a5fa; font-weight:700;'>
                {road_pct:.0f}%
            </div>
            <div style='font-size:0.8rem; color:#93c5fd; margin-top:4px;'>
                of {shap_target} prediction explained by road network features
            </div>
        </div>""", unsafe_allow_html=True)

        for feat, val in zip(df_shap["feature"], df_shap[shap_col]):
            is_road = any(k in feat for k in ["road", "intersection", "betweenness", "degree"])
            color   = "#3b82f6" if is_road else "#334155"
            bar_w   = int(val / df_shap[shap_col].max() * 100)
            st.markdown(f"""
            <div style='margin-bottom:8px;'>
                <div style='display:flex; justify-content:space-between;
                            font-size:0.72rem; color:#64748b; margin-bottom:3px;'>
                    <span>{feat}</span>
                    <span style='font-family:Space Mono,monospace'>{val:.3f}</span>
                </div>
                <div style='background:#1e2433; border-radius:3px; height:4px;'>
                    <div style='background:{color}; width:{bar_w}%; 
                                height:4px; border-radius:3px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)