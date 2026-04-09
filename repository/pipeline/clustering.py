#!/usr/bin/env python3
"""
Generate fan segments for a specific client using K-Means clustering.

Usage: python3 clustering.py <client_name>
Example: python3 clustering.py 5wins
         python3 clustering.py allez_sports
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---- CLIENT CONFIGURATION ----
if len(sys.argv) < 2:
    print("Usage: python3 clustering.py <client_name>")
    print("Example: python3 clustering.py 5wins")
    sys.exit(1)

CLIENT = sys.argv[1]
REPO_PATH = Path(__file__).parent.parent / "output" / f"repository_signals_{CLIENT}.json"
OUTPUT_PATH = Path(__file__).parent.parent / "output" / f"fan_segments_{CLIENT}.json"

if not REPO_PATH.exists():
    print(f"ERROR: {REPO_PATH} not found")
    sys.exit(1)

# ---- LOAD AND PREPARE DATA ----
print(f"Loading {CLIENT} signals from {REPO_PATH}...")
with open(REPO_PATH) as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Loaded {len(df)} records")

# Filter: remove reddit posts tagged only as 'general'
mask_drop = (df["source"] == "reddit") & (df["sports"].apply(lambda x: x == ["general"]))
df_filtered = df[~mask_drop].reset_index(drop=True)
print(f"After filtering: {len(df_filtered)} records")

# Map sentiment to numeric
sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
df_filtered["sentiment_encoded"] = df_filtered["sentiment"].map(sentiment_map)

# One-hot encode behavioral_pathway and priority_signal
df_encoded = pd.get_dummies(df_filtered, columns=["behavioral_pathway", "priority_signal"])

# ---- FEATURE ENGINEERING ----
feature_cols = [
    "sentiment_score",
    "emotional_affinity_score",
    "confidence_score",
]

X = df_encoded[feature_cols].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- K-MEANS CLUSTERING ----
print("Running K-Means clustering (k=6)...")
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(X_scaled)
df_encoded["cluster"] = kmeans.labels_

# ---- LABEL CLUSTERS ----
cluster_labels = {
    0: "Superfan",
    1: "Emotionally Invested, Weak Signal",
    2: "Core Engaged Fan",
    3: "Casual Enthusiast",
    4: "Frustrated Loyalist",
    5: "Passive / Disengaged"
}

df_encoded["segment"] = df_encoded["cluster"].map(cluster_labels)

# ---- EXPORT ----
export_cols = [
    "record_id",
    "source",
    "sports",
    "sentiment_encoded",
    "sentiment_score",
    "emotional_affinity_score",
    "confidence_score",
    "cluster",
    "segment"
]

df_export = df_encoded[export_cols].copy()

with open(OUTPUT_PATH, "w") as f:
    json.dump(df_export.to_dict(orient="records"), f, indent=2)

print(f"\nDone. {len(df_export)} segments saved to {OUTPUT_PATH}")

# Print summary
print(f"\nSegment breakdown:")
for segment, count in df_export["segment"].value_counts().sort_index().items():
    print(f"  {segment}: {count}")
