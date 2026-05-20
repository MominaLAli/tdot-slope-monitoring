import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from pathlib import Path
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

# ── 1. Find all vert_disp files and extract dates ─────────────────────────────
insar_dir  = Path('data/raw/sentinel1_insar')
disp_files = sorted(insar_dir.rglob('*_vert_disp.tif'))
corr_files = sorted(insar_dir.rglob('*_corr.tif'))
print(f"Displacement files found: {len(disp_files)}")

def extract_dates(path):
    dates = re.findall(r'(\d{8})T', path.stem)
    return (dates[0], dates[1]) if len(dates) >= 2 else (None, None)

file_info = []
for dp in disp_files:
    cp = dp.parent / dp.name.replace('_vert_disp','_corr')
    d1, d2 = extract_dates(dp)
    if d1 and d2 and cp.exists():
        file_info.append({
            'disp_path': dp, 'corr_path': cp,
            'date1':     pd.to_datetime(d1),
            'date2':     pd.to_datetime(d2),
            'mid_date':  pd.to_datetime(d1) +
                         (pd.to_datetime(d2)-pd.to_datetime(d1))/2,
        })

file_info = sorted(file_info, key=lambda x: x['mid_date'])
print(f"Valid pairs: {len(file_info)}")
print(f"Date range: {file_info[0]['mid_date'].date()} → {file_info[-1]['mid_date'].date()}")

# ── 2. Load segments ──────────────────────────────────────────────────────────
segs = gpd.read_file('data/raw/roads/i40_corridor_utm.geojson')
with rasterio.open(file_info[0]['disp_path']) as src:
    insar_crs = src.crs
segs_buf = segs.to_crs(insar_crs).copy()
segs_buf['geometry'] = segs_buf.geometry.buffer(500)

# ── 3. Zonal mean ─────────────────────────────────────────────────────────────
def zonal_mean(geom, src, scale=1000.0):
    try:
        masked, _ = rio_mask(src, [geom], crop=True, nodata=np.nan)
        v = masked[0].flatten().astype(float)
        if src.nodata is not None:
            v[v == src.nodata] = np.nan
        v = v[~np.isnan(v)]
        v = v[np.abs(v) < 1e6]
        return float(np.mean(v) * scale) if len(v) >= 3 else np.nan
    except:
        return np.nan

# ── 4. Extract time series ────────────────────────────────────────────────────
print(f"\nExtracting time series: {len(segs_buf)} segments × {len(file_info)} pairs...")
print("This takes 5-10 minutes...")

records = []
for i, fi in enumerate(file_info):
    date_str = fi['mid_date'].strftime('%Y-%m-%d')
    with rasterio.open(fi['disp_path']) as d_src, \
         rasterio.open(fi['corr_path']) as c_src:
        for _, row in segs_buf.iterrows():
            g = row.geometry.__geo_interface__
            records.append({
                'segment_id': row['segment_id'],
                'date':       date_str,
                'disp_mm':    round(zonal_mean(g, d_src, 1000.0), 3),
                'coherence':  round(zonal_mean(g, c_src, 1.0),    3),
            })
    if (i+1) % 5 == 0:
        print(f"  Processed {i+1}/{len(file_info)} pairs...")

df_ts = pd.DataFrame(records)
df_ts.to_csv('data/processed/insar_features/insar_timeseries.csv', index=False)
print(f"\nTime series saved: {df_ts.shape}")

# ── 5. Compute trend per segment ──────────────────────────────────────────────
print("Computing displacement trends...")
trend_records = []
for sid, grp in df_ts.groupby('segment_id'):
    grp = grp.dropna(subset=['disp_mm']).sort_values('date')
    if len(grp) < 3:
        trend_records.append({
            'segment_id': sid, 'n_pairs': len(grp),
            'mean_disp_mm': np.nan, 'disp_trend_mm_per_month': np.nan,
            'disp_std_mm': np.nan,  'mean_coherence': np.nan,
            'max_disp_mm': np.nan,  'min_disp_mm': np.nan
        })
        continue
    days = (pd.to_datetime(grp['date']) -
            pd.to_datetime(grp['date'].iloc[0])).dt.days.values
    vals = grp['disp_mm'].values
    slope = np.polyfit(days, vals, 1)[0] * 30 if days[-1] > 0 else 0.0
    trend_records.append({
        'segment_id':              sid,
        'n_pairs':                 len(grp),
        'mean_disp_mm':            round(float(grp['disp_mm'].mean()),  3),
        'disp_trend_mm_per_month': round(float(slope),                  4),
        'disp_std_mm':             round(float(grp['disp_mm'].std()),   3),
        'mean_coherence':          round(float(grp['coherence'].mean()),3),
        'max_disp_mm':             round(float(grp['disp_mm'].max()),   3),
        'min_disp_mm':             round(float(grp['disp_mm'].min()),   3),
    })

df_trend = pd.DataFrame(trend_records)
df_trend.to_csv('data/processed/insar_features/insar_features_extended.csv', index=False)

print(f"\nExtended InSAR features: {df_trend.shape}")
print(f"  Mean trend:       {df_trend.disp_trend_mm_per_month.mean():.3f} mm/month")
print(f"  Most negative:    {df_trend.disp_trend_mm_per_month.min():.3f} mm/month "
      f"(seg {df_trend.loc[df_trend.disp_trend_mm_per_month.idxmin(),'segment_id']})")
print(f"  Mean coherence:   {df_trend.mean_coherence.mean():.3f}")
print(f"\nTop 5 by subsidence rate:")
print(df_trend.nsmallest(5,'disp_trend_mm_per_month')[
    ['segment_id','disp_trend_mm_per_month','mean_disp_mm','mean_coherence','n_pairs']
].to_string(index=False))
print("\nTime series extraction complete.")
