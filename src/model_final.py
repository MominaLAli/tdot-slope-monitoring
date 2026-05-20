import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── 1. Load all data sources ──────────────────────────────────────────────────
print("Loading all data sources...")
terrain = pd.read_csv('data/processed/terrain_features/terrain_features.csv')
insar   = pd.read_csv('data/processed/insar_features/insar_features.csv')
geology = pd.read_csv('data/raw/geology/geology_per_segment.csv')
traffic = pd.read_csv('data/raw/traffic/aadt_per_segment.csv')

# Merge all
df = terrain.copy()
df = df.merge(insar,   on='segment_id', how='left')
df = df.merge(geology, on='segment_id', how='left')
df = df.merge(traffic, on='segment_id', how='left')

print(f"  Combined table: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ── 2. Encode geology ─────────────────────────────────────────────────────────
rock_risk = {
    'metamorphic_crystalline': 4,  # highest rockfall risk
    'clastic_sedimentary':     3,  # landslide prone
    'mixed_sedimentary':       2,
    'carbonate':               2,  # dissolution/sinkhole risk
    'unconsolidated':          3,  # slope failure risk
    'other':                   1,
}
df['rock_risk_score'] = df['rock_class'].map(rock_risk).fillna(1)
df['geology_enc']     = LabelEncoder().fit_transform(df['rock_class'].fillna('other'))
df['landcover_enc']   = LabelEncoder().fit_transform(
    np.where(df['mean_elevation_m'] > 600, 'forest_high',
    np.where(df['mean_elevation_m'] > 300, 'forest_mixed', 'developed'))
)

# ── 3. Compute labels from real data ──────────────────────────────────────────
print("\nComputing risk labels from real data...")

def norm(s):
    s = pd.to_numeric(s, errors='coerce').fillna(0)
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series(np.zeros(len(s)))

label_score = (
    0.25 * norm(df['mean_slope_deg'])         +
    0.20 * norm(df['elev_range_m'])           +
    0.20 * norm(abs(df['mean_disp_mm']))      +
    0.15 * norm(df['terrain_roughness'])      +
    0.10 * norm(1 - df['mean_coherence'].fillna(0.5)) +
    0.10 * norm(df['rock_risk_score'])
) + np.random.normal(0, 0.02, len(df))

df['known_unstable'] = (label_score >= np.percentile(label_score, 75)).astype(int)
print(f"  Unstable segments: {df['known_unstable'].sum()} / {len(df)}")
print(f"  Rock class breakdown of unstable segments:")
print(df[df['known_unstable']==1]['rock_class'].value_counts().to_string())

# ── 4. Feature matrix ─────────────────────────────────────────────────────────
features = [
    # Terrain
    'mean_slope_deg', 'max_slope_deg', 'elev_range_m',
    'terrain_roughness', 'mean_curvature', 'stream_proximity_pct',
    # InSAR
    'mean_disp_mm', 'max_disp_mm', 'std_disp_mm',
    'mean_coherence', 'low_coh_pct',
    # Geology
    'geology_enc', 'rock_risk_score',
    # Traffic
    'AADT_2022', 'truck_pct', 'truck_AADT',
    # Land cover
    'landcover_enc',
]
X = df[features].fillna(0)
y = df['known_unstable']

# ── 5. Train three models ─────────────────────────────────────────────────────
print("\nTraining models...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Random Forest':   RandomForestClassifier(n_estimators=300, max_depth=8,
                           class_weight='balanced', random_state=42),
    'Gradient Boost':  GradientBoostingClassifier(n_estimators=200, max_depth=4,
                           learning_rate=0.05, random_state=42),
    'Logistic Reg':    LogisticRegression(class_weight='balanced',
                           max_iter=1000, random_state=42),
}

best_model = None
best_auc   = 0

for name, model in models.items():
    if name == 'Logistic Reg':
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = X
    cv = cross_val_score(model, Xs, y, cv=skf, scoring='roc_auc')
    print(f"  {name:20s}  ROC-AUC: {cv.mean():.3f} ± {cv.std():.3f}")
    if cv.mean() > best_auc:
        best_auc   = cv.mean()
        best_model = (name, model, X)

print(f"\n  Best model: {best_model[0]}")
best_model[1].fit(best_model[2], y)
df['rf_prob'] = best_model[1].predict_proba(best_model[2])[:, 1]

# ── 6. Composite risk score ───────────────────────────────────────────────────
terrain_susc  = (0.4*norm(df['mean_slope_deg']) +
                 0.3*norm(df['elev_range_m'])    +
                 0.3*norm(df['terrain_roughness']))
insar_signal  = (0.5*norm(abs(df['mean_disp_mm'])) +
                 0.3*norm(df['std_disp_mm'])        +
                 0.2*norm(1 - df['mean_coherence'].fillna(0.5)))
road_exposure = (0.6*norm(df['AADT_2022'].astype(float)) +
                 0.4*norm(df['truck_AADT'].astype(float)))
geo_factor    = norm(df['rock_risk_score'].astype(float))

df['risk_score'] = (
    0.35 * df['rf_prob']   +
    0.25 * terrain_susc    +
    0.20 * insar_signal    +
    0.12 * road_exposure   +
    0.08 * geo_factor
).round(4)

df['risk_category'] = pd.cut(
    df['risk_score'],
    bins=[0, 0.25, 0.50, 0.75, 1.01],
    labels=['Low','Moderate','High','Critical']
)

# ── 7. Save ───────────────────────────────────────────────────────────────────
df.to_csv('data/processed/model_table/full_feature_table.csv', index=False)

top10 = df.nlargest(10, 'risk_score')[
    ['segment_id','risk_score','risk_category','mean_slope_deg',
     'elev_range_m','mean_disp_mm','mean_coherence',
     'rock_class','AADT_2022','truck_pct']
].round(3)
top10.to_csv('outputs/tables/top_risk_segments.csv', index=False)

if hasattr(best_model[1], 'feature_importances_'):
    pd.Series(best_model[1].feature_importances_, index=features)\
      .sort_values(ascending=False)\
      .to_csv('outputs/tables/feature_importance.csv', header=['importance'])

print(f"\nRisk category distribution:")
print(df['risk_category'].value_counts().sort_index().to_string())
print(f"\nTop 5 highest-risk segments:")
print(top10.head(5).to_string(index=False))
print(f"\nData sources used:")
print(f"  Terrain:  USGS 3DEP 30m DEM (real)")
print(f"  InSAR:    Sentinel-1 ASF HyP3 — 5 pairs Dec2023-Jan2024 (real)")
print(f"  Geology:  USGS State Geology Map Tennessee (real)")
print(f"  Traffic:  FHWA HPMS 2022 (real)")
print(f"  Labels:   Terrain+InSAR+Geology composite (proxy — pending USMP)")
print("\nPhase 8 (final model) complete.")
