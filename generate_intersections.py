"""
URBAN-AIRNet | generate_intersections.py
=========================================
Extracts high-traffic road intersections from OpenStreetMap
for Chennai and saves them as a CSV for the dashboard map.

Run once:  python generate_intersections.py
Output  :  dashboard/data/intersections.csv
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CHENNAI_BBOX   = (13.0, 80.1, 13.35, 80.35)   # south, west, north, east
OUTPUT_PATH    = "dashboard/data/intersections.csv"
TOP_PERCENTILE = 0.75   # keep top 25% busiest intersections
MIN_DEGREE     = 3      # minimum roads meeting at intersection

# Road type weights (motorisation score)
ROAD_WEIGHTS = {
    "motorway": 1.0, "trunk": 0.9, "primary": 0.8,
    "secondary": 0.6, "tertiary": 0.4, "residential": 0.2,
    "unclassified": 0.1, "service": 0.1,
}
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 55)
print("  URBAN-AIRNet — Intersection Extractor")
print("=" * 55)

# Step 1: Download Chennai road graph
print("\n[OSM] Downloading Chennai road network...")
south, west, north, east = CHENNAI_BBOX
G = ox.graph_from_bbox(
    bbox=(north, south, east, west),
    network_type="drive",
    simplify=True,
)
print(f"[OSM] Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")

# Step 2: Compute betweenness centrality
print("\n[GRAPH] Computing betweenness centrality (approx)...")
G_undirected = G.to_undirected()
n = G.number_of_nodes()
k = min(500, n)
bc = nx.betweenness_centrality(G_undirected, k=k, normalized=True, weight="length")
print(f"[GRAPH] Centrality computed for {len(bc):,} nodes")

# Step 3: Extract node features
print("\n[NODES] Extracting intersection features...")
records = []
for node_id, data in G.nodes(data=True):
    lat = data.get("y", 0)
    lon = data.get("x", 0)
    degree = G.degree(node_id)

    # Skip low-degree nodes (dead ends, simple bends)
    if degree < MIN_DEGREE:
        continue

    # Get road types for edges connected to this node
    edge_types = []
    for _, _, edata in G.edges(node_id, data=True):
        hw = edata.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        edge_types.append(str(hw).lower())

    # Road type score
    road_score = np.mean([ROAD_WEIGHTS.get(t, 0.1) for t in edge_types])

    # Skip purely residential/footpath intersections
    if road_score < 0.15:
        continue

    centrality = bc.get(node_id, 0)

    records.append({
        "node_id":              node_id,
        "lat":                  round(lat, 5),
        "lon":                  round(lon, 5),
        "degree":               degree,
        "betweenness":          round(centrality, 8),
        "road_type_score":      round(road_score, 4),
        "dominant_road_type":   max(set(edge_types), key=edge_types.count),
    })

df = pd.DataFrame(records)
print(f"[NODES] Intersections after filtering: {len(df):,}")

# Step 4: Keep only top 25% by betweenness (busiest intersections)
threshold = df["betweenness"].quantile(TOP_PERCENTILE)
df = df[df["betweenness"] >= threshold].copy()
print(f"[FILTER] High-traffic intersections (top 25%): {len(df):,}")

# Step 5: Normalize features for model input
df["intersection_density"] = df["degree"] / (df["degree"].max())
df["node_degree_mean"]     = df["degree"]
df["betweenness_centrality"] = df["betweenness"]

# Step 6: Add average met conditions (from CPCB data mean)
# These will be overridden in app.py with latest real values
df["AT_(°C)"]      = 28.5    # Chennai annual mean temp
df["RH_(%)"]       = 72.0    # Chennai annual mean humidity
df["WS_(m/s)"]     = 2.1     # Chennai annual mean wind speed
df["WD_(deg)"]     = 225.0   # SW monsoon dominant direction
df["RF_(mm)"]      = 0.0
df["SR_(W/mt2)"]   = 220.0
df["BP_(mmHg)"]    = 1010.0
df["NOx_(ppb)"]    = 35.0    # estimated from CPCB mean
df["NO_(µg/m³)"]   = 18.0
df["SO2_(µg/m³)"]  = 8.0
df["CO_(mg/m³)"]   = 1.2
df["NH3_(µg/m³)"]  = 12.0
df["PM2.5_(µg/m³)"] = 45.0
df["PM10_(µg/m³)"] = 80.0
df["Benzene_(µg/m³)"] = 2.1
df["Toluene_(µg/m³)"] = 4.5
df["Ozone_(µg/m³)"]   = 28.0

# Step 7: Scale NOx by road type score (more traffic = more NOx)
df["NOx_(ppb)"]    = df["NOx_(ppb)"]    * (0.5 + df["road_type_score"])
df["NO_(µg/m³)"]   = df["NO_(µg/m³)"]  * (0.5 + df["road_type_score"])
df["CO_(mg/m³)"]   = df["CO_(mg/m³)"]  * (0.5 + df["road_type_score"])
df["PM2.5_(µg/m³)"] = df["PM2.5_(µg/m³)"] * (0.5 + df["road_type_score"])

# Step 8: Save
os.makedirs("dashboard/data", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[SAVE] → {OUTPUT_PATH}")
print(f"[DONE] {len(df):,} high-traffic intersections saved!")
print(df[["lat","lon","degree","betweenness","road_type_score","dominant_road_type"]].head(10).to_string())