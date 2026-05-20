import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
import richdem as rd
import pandas as pd
from scipy.ndimage import generic_filter
import tempfile, os, warnings
warnings.filterwarnings('ignore')

print("Extracting terrain features for I-75 corridor...")

segs = gpd.read_file('data/raw/roads/i75_corridor_utm.geojson')

with rasterio.open('data/raw/dem_lidar/i75_dem_clipped.tif') as src:
    dem_array = src.read(1).astype(float)
    profile   = src.profile
    transform = src.transform
    res_x     = abs(transform.a)
    res_y     = abs(transform.e)
    bounds    = src.bounds

dem_array[dem_array < -9000] = np.nan
print(f"  Resolution: {res_x:.1f} x {res_y:.1f} m")

# Terrain attributes
rda = rd.rdarray(dem_array, no_data=np.nan)
rda.geotransform = (bounds.left, res_x, 0, bounds.top, 0, -res_y)
slope_arr    = np.array(rd.TerrainAttribute(rda, attrib='slope_degrees'))
curve_arr    = np.array(rd.TerrainAttribute(rda, attrib='curvature'))
roughness_arr = generic_filter(dem_array, np.nanstd, size=5)
print(f"  Slope range: {np.nanmin(slope_arr):.1f}° – {np.nanmax(slope_arr):.1f}°")

# Write temp rasters
def write_temp(arr, profile):
    p = profile.copy()
    p.update(dtype=rasterio.float32, count=1, nodata=-9999)
    tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    a = np.where(np.isnan(arr), -9999, arr).astype(np.float32)
    with rasterio.open(tmp.name, 'w', **p) as dst:
        dst.write(a, 1)
    return tmp.name

dem_path   = 'data/raw/dem_lidar/i75_dem_clipped.tif'
slope_path = write_temp(slope_arr,     profile)
curve_path = write_temp(curve_arr,     profile)
rough_path = write_temp(roughness_arr, profile)

def zonal(geom_dict, src):
    try:
        masked, _ = rio_mask(src, [geom_dict], crop=True, nodata=np.nan)
        v = masked[0].flatten()
        v = v[(~np.isnan(v)) & (v > -9000)]
        if len(v) == 0:
            return np.nan, np.nan, np.nan, np.nan
        return float(np.mean(v)),float(np.max(v)),float(np.std(v)),float(np.max(v)-np.min(v))
    except:
        return np.nan, np.nan, np.nan, np.nan

segs_buf = segs.to_crs(profile['crs']).copy()
segs_buf['geometry'] = segs_buf.geometry.buffer(500)

print(f"Processing {len(segs_buf)} segments...")
records = []

with rasterio.open(dem_path)   as d_src, \
     rasterio.open(slope_path) as s_src, \
     rasterio.open(curve_path) as c_src, \
     rasterio.open(rough_path) as r_src:

    for idx, row in segs_buf.iterrows():
        g = row.geometry.__geo_interface__
        e_mean,e_max,e_std,e_range = zonal(g, d_src)
        s_mean,s_max,s_std,_       = zonal(g, s_src)
        c_mean,_,_,_               = zonal(g, c_src)
        r_mean,_,_,_               = zonal(g, r_src)

        try:
            masked, _ = rio_mask(d_src, [g], crop=True, nodata=np.nan)
            v = masked[0].flatten()
            v = v[(~np.isnan(v)) & (v > -9000)]
            p10 = np.percentile(v, 10) if len(v) > 0 else np.nan
            stream_pct = float(np.mean(v <= p10)*100) if len(v) > 0 else np.nan
        except:
            stream_pct = np.nan

        records.append({
            'segment_id':           row['segment_id'],
            'road_name':            'I-75',
            'mean_elevation_m':     round(e_mean,  1) if not np.isnan(e_mean)  else np.nan,
            'elev_range_m':         round(e_range, 1) if not np.isnan(e_range) else np.nan,
            'mean_slope_deg':       round(s_mean,  2) if not np.isnan(s_mean)  else np.nan,
            'max_slope_deg':        round(s_max,   2) if not np.isnan(s_max)   else np.nan,
            'std_slope_deg':        round(s_std,   2) if not np.isnan(s_std)   else np.nan,
            'mean_curvature':       round(c_mean,  4) if not np.isnan(c_mean)  else np.nan,
            'terrain_roughness':    round(r_mean,  2) if not np.isnan(r_mean)  else np.nan,
            'stream_proximity_pct': round(stream_pct,1) if not np.isnan(stream_pct) else np.nan,
        })

for p in [slope_path, curve_path, rough_path]:
    os.unlink(p)

df = pd.DataFrame(records)
df.to_csv('data/processed/terrain_features/terrain_features_i75.csv', index=False)

print(f"\nI-75 terrain features saved: {df.shape}")
print(f"  Mean slope: {df.mean_slope_deg.mean():.1f}°")
print(f"  Max slope:  {df.max_slope_deg.max():.1f}°")
print(f"  Elev range: {df.elev_range_m.max():.0f}m max")
print(f"\nSample:")
print(df.head(5).to_string(index=False))
print("\nI-75 terrain features complete.")
