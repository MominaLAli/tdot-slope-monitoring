import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load and merge ─────────────────────────────────────────────────────────
print("Loading USGS Tennessee geology...")
geo   = gpd.read_file('data/raw/geology/TN_geol_poly.shp')
units = pd.read_csv('data/raw/geology/TN_units.csv', encoding='latin1')
geo.columns   = geo.columns.str.lower()
units.columns = units.columns.str.lower()
geo = geo.merge(units, on='unit_link', how='left')

# ── 2. Classify rock type ─────────────────────────────────────────────────────
def classify_rock(row):
    combined = ' '.join([
        str(row.get('rocktype1','')),
        str(row.get('rocktype2','')),
        str(row.get('unit_name',''))
    ]).lower()
    if any(x in combined for x in ['limestone','dolomite','carbonate']):
        return 'carbonate'
    elif any(x in combined for x in ['sandstone','shale','mudstone','siltstone','clastic']):
        return 'clastic_sedimentary'
    elif any(x in combined for x in ['granite','gneiss','schist','metamorphic','phyllite','quartzite']):
        return 'metamorphic_crystalline'
    elif any(x in combined for x in ['alluvium','gravel','sand','unconsolidated']):
        return 'unconsolidated'
    elif any(x in combined for x in ['chert','conglomerate']):
        return 'mixed_sedimentary'
    else:
        return 'other'

geo['rock_class'] = geo.apply(classify_rock, axis=1)

# ── 3. Spatial join ───────────────────────────────────────────────────────────
print("Joining geology to road segments...")
segs     = gpd.read_file('data/processed/corridor/segments.geojson')
segs_buf = segs.copy()
segs_buf['geometry'] = segs.geometry.buffer(0.01)

geo_clean = geo[['sgmc_label','rocktype1','unit_name','rock_class','geometry']].copy()
geo_clean = geo_clean[geo_clean.geometry.is_valid].reset_index(drop=True)

joined = gpd.sjoin(segs_buf, geo_clean, how='left', predicate='intersects')

# Safe mode function
def safe_mode(x):
    m = x.dropna()
    if len(m) == 0:
        return 'other'
    counts = m.value_counts()
    return counts.index[0]

agg = joined.groupby('segment_id').agg(
    rock_class = ('rock_class', safe_mode),
    rocktype1  = ('rocktype1',  safe_mode),
    unit_name  = ('unit_name',  safe_mode),
).reset_index()

print(f"  Segments assigned: {len(agg)}")
print(f"\n  Rock class along I-40 corridor:")
print(agg['rock_class'].value_counts().to_string())
print(f"\nSample:")
print(agg.head(10).to_string(index=False))

# ── 4. Save ───────────────────────────────────────────────────────────────────
agg.to_csv('data/raw/geology/geology_per_segment.csv', index=False)
print("\nPhase 6 (geology) complete.")
