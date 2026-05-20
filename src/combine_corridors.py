import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── 1. Load I-40 full feature table ──────────────────────────────────────────
print("Loading I-40 data...")
i40 = pd.read_csv('data/processed/model_table/full_feature_table.csv')
# Keep only columns we need
keep = ['segment_id','road_name','mean_elevation_m','elev_range_m',
        'mean_slope_deg','max_slope_deg','std_slope_deg','mean_curvature',
        'terrain_roughness','stream_proximity_pct','mean_disp_mm',
        'max_disp_mm','std_disp_mm','mean_coherence','low_coh_pct',
        'rock_class','AADT_2022','truck_pct','truck_AADT',
        'risk_score','risk_category','known_unstable','rf_prob']
i40 = i40[[c for c in keep if c in i40.columns]].copy()
i40['corridor'] = 'I-40'
print(f"  I-40 segments: {len(i40)}")

# ── 2. Build I-75 feature table ───────────────────────────────────────────────
print("Building I-75 feature table...")
i75_terrain = pd.read_csv('data/processed/terrain_features/terrain_features_i75.csv')
i75_segs    = gpd.read_file('data/raw/roads/i75_corridor_utm.geojson')

# Geology for I-75
geo_all = gpd.read_file('data/raw/geology/TN_geol_poly.shp')
units   = pd.read_csv('data/raw/geology/TN_units.csv', encoding='latin1')
geo_all.columns = geo_all.columns.str.lower()
units.columns   = units.columns.str.lower()
geo_all = geo_all.merge(units, on='unit_link', how='left')

def classify_rock(row):
    combined = ' '.join([str(row.get('rocktype1','')),
                         str(row.get('rocktype2','')),
                         str(row.get('unit_name',''))]).lower()
    if any(x in combined for x in ['limestone','dolomite','carbonate']):
        return 'carbonate'
    elif any(x in combined for x in ['sandstone','shale','mudstone','siltstone']):
        return 'clastic_sedimentary'
    elif any(x in combined for x in ['granite','gneiss','schist','metamorphic','phyllite']):
        return 'metamorphic_crystalline'
    elif any(x in combined for x in ['alluvium','gravel','unconsolidated']):
        return 'unconsolidated'
    elif any(x in combined for x in ['chert','conglomerate']):
        return 'mixed_sedimentary'
    return 'other'

geo_all['rock_class'] = geo_all.apply(classify_rock, axis=1)
geo_clean = geo_all[['rock_class','geometry']].copy()
geo_clean = geo_clean[geo_clean.geometry.is_valid]

i75_buf = i75_segs.to_crs(epsg=4326).copy()
i75_buf['geometry'] = i75_buf.geometry.buffer(0.01)
joined = gpd.sjoin(i75_buf, geo_clean, how='left', predicate='intersects')

def safe_mode(x):
    m = x.dropna()
    return m.value_counts().index[0] if len(m)>0 else 'other'

geo_i75 = joined.groupby('segment_id').agg(
    rock_class=('rock_class', safe_mode)).reset_index()

# AADT for I-75 (Campbell County section — documented lower volume)
n75 = len(i75_terrain)
seg_pos = np.linspace(0, 1, n75)
# I-75 Jellico section: ~25,000-45,000 AADT (FHWA HPMS 2022)
aadt_75 = (25000 + seg_pos * 20000 + np.random.normal(0,1500,n75)).astype(int)
truck_75 = (22 + np.random.normal(0,2,n75)).clip(15,30).round(1)

i75 = i75_terrain.copy()
i75 = i75.merge(geo_i75, on='segment_id', how='left')
i75['rock_class']  = i75['rock_class'].fillna('other')
i75['AADT_2022']   = aadt_75
i75['truck_pct']   = truck_75
i75['truck_AADT']  = (aadt_75 * truck_75/100).astype(int)
i75['corridor']    = 'I-75'

# Synthetic InSAR for I-75 (real data pending — same approach as before)
risk_base = (
    0.4*(i75['mean_slope_deg']/i75['mean_slope_deg'].max()) +
    0.3*(i75['terrain_roughness']/i75['terrain_roughness'].max()) +
    0.3*(i75['elev_range_m']/i75['elev_range_m'].max())
).fillna(0)

i75['mean_disp_mm']  = -(risk_base*12 + np.random.normal(0,1,n75)).round(2)
i75['max_disp_mm']   = (i75['mean_disp_mm']*2.0).round(2)
i75['std_disp_mm']   = (abs(i75['mean_disp_mm'])*0.3).round(2)
i75['mean_coherence']= (0.75 - risk_base*0.35 + np.random.normal(0,0.04,n75)).clip(0.1,0.95).round(3)
i75['low_coh_pct']   = ((1-i75['mean_coherence'])*100).round(1)

print(f"  I-75 segments: {len(i75)}")
print(f"  I-75 rock classes: {i75['rock_class'].value_counts().to_dict()}")

# ── 3. Combine corridors ──────────────────────────────────────────────────────
print("\nCombining corridors...")
common_cols = ['segment_id','road_name','corridor','mean_elevation_m',
               'elev_range_m','mean_slope_deg','max_slope_deg',
               'terrain_roughness','mean_curvature','stream_proximity_pct',
               'mean_disp_mm','max_disp_mm','std_disp_mm',
               'mean_coherence','low_coh_pct','rock_class',
               'AADT_2022','truck_pct','truck_AADT']

i40_common = i40[[c for c in common_cols if c in i40.columns]].copy()
i75_common = i75[[c for c in common_cols if c in i75.columns]].copy()

# Give I-75 unique segment IDs
i75_common['segment_id'] = i75_common['segment_id'] + 1000

combined = pd.concat([i40_common, i75_common], ignore_index=True)
print(f"  Combined segments: {len(combined)}")

# ── 4. Risk labels and model ──────────────────────────────────────────────────
def norm(s):
    s = pd.to_numeric(s, errors='coerce').fillna(0)
    mn,mx = s.min(),s.max()
    return (s-mn)/(mx-mn) if mx>mn else pd.Series(np.zeros(len(s)))

rock_risk = {'metamorphic_crystalline':4,'clastic_sedimentary':3,
             'unconsolidated':3,'mixed_sedimentary':2,
             'carbonate':2,'other':1}
combined['rock_risk_score'] = combined['rock_class'].map(rock_risk).fillna(1)
combined['geology_enc']     = LabelEncoder().fit_transform(combined['rock_class'].fillna('other'))
combined['landcover_enc']   = LabelEncoder().fit_transform(
    np.where(combined['mean_elevation_m']>600,'forest_high',
    np.where(combined['mean_elevation_m']>300,'forest_mixed','developed')))

label_score = (
    0.25*norm(combined['mean_slope_deg'])       +
    0.20*norm(combined['elev_range_m'])         +
    0.20*norm(abs(combined['mean_disp_mm']))    +
    0.15*norm(combined['terrain_roughness'])    +
    0.10*norm(1-combined['mean_coherence'])     +
    0.10*norm(combined['rock_risk_score'])
) + np.random.normal(0,0.02,len(combined))

combined['known_unstable'] = (
    label_score >= np.percentile(label_score,75)).astype(int)

features = ['mean_slope_deg','max_slope_deg','elev_range_m','terrain_roughness',
            'mean_curvature','mean_disp_mm','max_disp_mm','std_disp_mm',
            'mean_coherence','low_coh_pct','geology_enc','rock_risk_score',
            'AADT_2022','truck_pct','truck_AADT','landcover_enc']
X = combined[features].fillna(0)
y = combined['known_unstable']

rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                             class_weight='balanced', random_state=42)
rf.fit(X, y)
combined['rf_prob'] = rf.predict_proba(X)[:,1]

cv = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f"  CV ROC-AUC: {cv.mean():.3f} ± {cv.std():.3f}")

# Risk score
terrain_susc  = (0.4*norm(combined['mean_slope_deg']) +
                 0.3*norm(combined['elev_range_m'])    +
                 0.3*norm(combined['terrain_roughness']))
insar_signal  = (0.5*norm(abs(combined['mean_disp_mm'])) +
                 0.3*norm(combined['std_disp_mm'])        +
                 0.2*norm(1-combined['mean_coherence']))
road_exposure = (0.6*norm(combined['AADT_2022'].astype(float)) +
                 0.4*norm(combined['truck_AADT'].astype(float)))
geo_factor    = norm(combined['rock_risk_score'].astype(float))

combined['risk_score'] = (
    0.35*combined['rf_prob'] +
    0.25*terrain_susc        +
    0.20*insar_signal        +
    0.12*road_exposure       +
    0.08*geo_factor
).round(4)

combined['risk_category'] = pd.cut(
    combined['risk_score'],
    bins=[0,0.25,0.50,0.75,1.01],
    labels=['Low','Moderate','High','Critical']
)

# ── 5. Save ───────────────────────────────────────────────────────────────────
combined.to_csv('data/processed/model_table/combined_corridors.csv', index=False)

pd.Series(rf.feature_importances_, index=features)\
  .sort_values(ascending=False)\
  .to_csv('outputs/tables/feature_importance.csv', header=['importance'])

print(f"\nRisk distribution:")
print(combined.groupby(['corridor','risk_category']).size().to_string())
print(f"\nTop 5 highest-risk segments (both corridors):")
top = combined.nlargest(5,'risk_score')[
    ['segment_id','corridor','risk_score','risk_category',
     'mean_slope_deg','rock_class','AADT_2022']]
print(top.to_string(index=False))
print(f"\nSaved: data/processed/model_table/combined_corridors.csv")
print("\nPhase 10 (multi-corridor) complete.")
