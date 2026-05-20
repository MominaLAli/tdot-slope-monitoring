import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── 1. Find all vert_disp and corr files ─────────────────────────────────────
insar_dir = Path('data/raw/sentinel1_insar')
disp_files = sorted(insar_dir.rglob('*_vert_disp.tif'))
corr_files  = sorted(insar_dir.rglob('*_corr.tif'))

print(f"Displacement files: {len(disp_files)}")
print(f"Coherence files:    {len(corr_files)}")

# ── 2. Load segments (reproject to match InSAR CRS) ──────────────────────────
segs = gpd.read_file('data/raw/roads/i40_corridor_utm.geojson')

# Check InSAR CRS
with rasterio.open(disp_files[0]) as src:
    insar_crs = src.crs
    print(f"InSAR CRS: {insar_crs}")

segs_insar = segs.to_crs(insar_crs)
segs_buf   = segs_insar.copy()
segs_buf['geometry'] = segs_insar.geometry.buffer(500)

# ── 3. Zonal stats function ───────────────────────────────────────────────────
def zonal(geom, src, scale=1.0):
    try:
        masked, _ = rio_mask(src, [geom], crop=True, nodata=np.nan)
        v = masked[0].flatten().astype(float)
        v[v == src.nodata] = np.nan
        v = v[~np.isnan(v)]
        v = v[np.abs(v) < 1e10]   # remove outliers
        if len(v) < 5:
            return np.nan, np.nan, np.nan, np.nan
        v = v * scale
        return float(np.mean(v)), float(np.max(v)), float(np.min(v)), float(np.std(v))
    except:
        return np.nan, np.nan, np.nan, np.nan

# ── 4. Extract per-pair features ──────────────────────────────────────────────
print(f"\nExtracting InSAR features for {len(segs_buf)} segments across {len(disp_files)} pairs...")

all_pair_records = []

for dp, cp in zip(disp_files, corr_files):
    pair_name = dp.stem.replace('_vert_disp','')
    dates = pair_name[5:13] + '_' + pair_name[22:30]
    print(f"  Processing pair: {dates}")

    pair_records = []
    with rasterio.open(dp) as d_src, rasterio.open(cp) as c_src:
        for idx, row in segs_buf.iterrows():
            g = row.geometry.__geo_interface__
            # Convert meters to mm (vert_disp is in meters)
            d_mean, d_max, d_min, d_std = zonal(g, d_src, scale=1000.0)
            c_mean, c_max, c_min, c_std = zonal(g, c_src, scale=1.0)

            pair_records.append({
                'segment_id':  row['segment_id'],
                'pair':        dates,
                'disp_mean_mm': round(d_mean, 3) if not np.isnan(d_mean) else np.nan,
                'disp_max_mm':  round(d_max,  3) if not np.isnan(d_max)  else np.nan,
                'disp_std_mm':  round(d_std,  3) if not np.isnan(d_std)  else np.nan,
                'coherence':    round(c_mean, 3) if not np.isnan(c_mean) else np.nan,
                'low_coh_pct':  round(float(np.mean(
                    [1 if c_mean < 0.3 else 0])) * 100, 1),
            })

    all_pair_records.extend(pair_records)

df_pairs = pd.DataFrame(all_pair_records)
df_pairs.to_csv('data/processed/insar_features/insar_per_pair.csv', index=False)
print(f"\nPer-pair table saved: {df_pairs.shape}")

# ── 5. Aggregate across all pairs per segment ─────────────────────────────────
print("Aggregating across pairs...")
agg = df_pairs.groupby('segment_id').agg(
    mean_disp_mm     = ('disp_mean_mm', 'mean'),
    max_disp_mm      = ('disp_max_mm',  'min'),   # most negative = max subsidence
    std_disp_mm      = ('disp_std_mm',  'mean'),
    mean_coherence   = ('coherence',    'mean'),
    low_coh_pct      = ('low_coh_pct',  'mean'),
    n_pairs          = ('pair',         'count'),
).reset_index()

agg = agg.round(3)
agg.to_csv('data/processed/insar_features/insar_features.csv', index=False)

print(f"\nAggregated InSAR features saved: {agg.shape}")
print(f"\nDisplacement stats (real satellite data):")
print(f"  Mean displacement: {agg.mean_disp_mm.mean():.3f} mm")
print(f"  Max displacement:  {agg.max_disp_mm.min():.3f} mm")
print(f"  Mean coherence:    {agg.mean_coherence.mean():.3f}")
print(f"\nSample (top 5 by displacement):")
print(agg.nsmallest(5,'max_disp_mm')[
    ['segment_id','mean_disp_mm','max_disp_mm','mean_coherence','low_coh_pct']
].to_string(index=False))
print("\nInSAR feature extraction complete.")
