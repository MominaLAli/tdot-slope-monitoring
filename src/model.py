import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── 1. Load terrain features ──────────────────────────────────────────────────
print("Loading terrain features...")
df = pd.read_csv('data/processed/terrain_features/terrain_features.csv')

# ── 2. Synthetic InSAR features ───────────────────────────────────────────────
print("Generating synthetic InSAR features...")
risk_base = (
    0.4 * (df['mean_slope_deg']    / df['mean_slope_deg'].max()) +
    0.3 * (df['terrain_roughness'] / df['terrain_roughness'].max()) +
    0.2 * (df['elev_range_m']      / df['elev_range_m'].max()) +
    0.1 * np.random.random(len(df))
).fillna(0)

df['mean_disp_mm']     = -(risk_base * 15 + np.random.normal(0, 1, len(df))).round(2)
df['max_disp_mm']      = (df['mean_disp_mm'] * 2.1 + np.random.normal(0, 0.5, len(df))).round(2)
df['mean_coherence']   = (0.8 - risk_base * 0.4 + np.random.normal(0, 0.05, len(df))).clip(0.1, 0.95).round(3)
df['low_coh_pct']      = ((1 - df['mean_coherence']) * 100).round(1)
df['movement_area_m2'] = (risk_base * 8000 + np.random.exponential(500, len(df))).round(0)

# ── 3. Geology, landcover, AADT ───────────────────────────────────────────────
print("Adding ancillary features...")
segs = gpd.read_file('data/raw/roads/i40_corridor_utm.geojson')
df['centroid_x'] = segs.geometry.centroid.x.values

df['geology_class'] = np.where(df['centroid_x'] < 400000, 'sedimentary',
                      np.where(df['centroid_x'] < 500000, 'ridge_valley', 'metamorphic'))

df['landcover_class'] = np.where(df['mean_elevation_m'] > 600, 'forest_high',
                        np.where(df['mean_elevation_m'] > 300, 'forest_mixed', 'developed'))

df['AADT'] = (80000 - risk_base * 40000 + np.random.normal(0, 3000, len(df))).clip(5000, 85000).astype(int)

# ── 4. Labels ─────────────────────────────────────────────────────────────────
print("Generating labels...")
label_score = (
    0.35 * (df['mean_slope_deg']    / df['mean_slope_deg'].max()) +
    0.25 * (df['elev_range_m']      / df['elev_range_m'].max()) +
    0.20 * (abs(df['mean_disp_mm']) / abs(df['mean_disp_mm']).max()) +
    0.20 * (df['terrain_roughness'] / df['terrain_roughness'].max())
).fillna(0) + np.random.normal(0, 0.05, len(df))

df['known_unstable'] = (label_score >= np.percentile(label_score, 75)).astype(int)
print(f"  Unstable segments: {df['known_unstable'].sum()} / {len(df)}")

# ── 5. Encode categoricals ────────────────────────────────────────────────────
le = LabelEncoder()
df['geology_enc']   = le.fit_transform(df['geology_class'])
df['landcover_enc'] = LabelEncoder().fit_transform(df['landcover_class'])

# ── 6. Train Random Forest ────────────────────────────────────────────────────
features = [
    'mean_slope_deg','max_slope_deg','elev_range_m','terrain_roughness',
    'mean_curvature','mean_disp_mm','max_disp_mm','mean_coherence',
    'low_coh_pct','movement_area_m2','geology_enc','landcover_enc','AADT'
]
X = df[features].fillna(0)
y = df['known_unstable']

rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             class_weight='balanced', random_state=42)
rf.fit(X, y)
df['rf_prob'] = rf.predict_proba(X)[:, 1]

cv = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f"  CV ROC-AUC: {cv.mean():.3f} ± {cv.std():.3f}")

# ── 7. Risk score ─────────────────────────────────────────────────────────────
def norm(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0

terrain_susc  = 0.5*norm(df['mean_slope_deg']) + 0.3*norm(df['elev_range_m']) + 0.2*norm(df['terrain_roughness'])
road_exposure = norm(df['AADT'].astype(float))
consequence   = norm(df['geology_enc'].astype(float))

df['risk_score'] = (
    0.40 * df['rf_prob'] +
    0.25 * terrain_susc +
    0.20 * road_exposure +
    0.15 * consequence
).round(4)

df['risk_category'] = pd.cut(
    df['risk_score'],
    bins=[0, 0.25, 0.50, 0.75, 1.01],
    labels=['Low','Moderate','High','Critical']
)

# ── 8. Save ───────────────────────────────────────────────────────────────────
df.to_csv('data/processed/model_table/full_feature_table.csv', index=False)

top10 = df.nlargest(10, 'risk_score')[
    ['segment_id','road_name','risk_score','risk_category',
     'mean_slope_deg','elev_range_m','mean_disp_mm','AADT']
].round(3)
top10.to_csv('outputs/tables/top_risk_segments.csv', index=False)

pd.Series(rf.feature_importances_, index=features)\
  .sort_values(ascending=False)\
  .to_csv('outputs/tables/feature_importance.csv', header=['importance'])

print(f"\nRisk category distribution:")
print(df['risk_category'].value_counts().sort_index().to_string())
print(f"\nTop 5 highest-risk segments:")
print(top10.head(5).to_string(index=False))
print("\nPhase 4 complete.")
