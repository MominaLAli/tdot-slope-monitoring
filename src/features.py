import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
import richdem as rd
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load segments and DEM ──────────────────────────────────────────────────
print("Loading segments and DEM...")
segs = gpd.read_file('data/raw/roads/i40_corridor_utm.geojson')
dem_path = 'data/raw/dem_lidar/dem_clipped.tif'

with rasterio.open(dem_path) as src:
    dem_crs = src.crs
    print(f"  DEM CRS: {dem_crs}")
    print(f"  Segments CRS: {segs.crs}")

# Reproject segments to match DEM if needed
if str(segs.crs) != str(dem_crs):
    segs = segs.to_crs(dem_crs)

# ── 2. Compute slope and curvature from DEM using richdem ────────────────────
print("Computing slope and curvature rasters...")
with rasterio.open(dem_path) as src:
    dem_array = src.read(1).astype(float)
    dem_array[dem_array < -9000] = np.nan
    profile = src.profile
    res = src.res[0]  # pixel size in meters

rda = rd.rdarray(dem_array, no_data=np.nan)
slope_arr  = rd.TerrainAttribute(rda, attrib='slope_degrees')
curve_arr  = rd.TerrainAttribute(rda, attrib='curvature')

# Terrain roughness = std of elevation in 3x3 window
from scipy.ndimage import generic_filter
roughness_arr = generic_filter(dem_array, np.nanstd, size=3)

print(f"  Slope range: {np.nanmin(slope_arr):.1f}° – {np.nanmax(slope_arr):.1f}°")

# ── 3. Extract zonal stats per segment buffer ─────────────────────────────────
print(f"Extracting features for {len(segs)} segments...")

def zonal_stats(geom, raster_src, arr):
    """Extract stats from array within geometry."""
    try:
        masked, _ = rio_mask(raster_src, [geom], crop=True, nodata=np.nan)
        vals = masked[0].flatten()
        vals = vals[~np.isnan(vals)]
        vals = vals[vals > -9000]
        if len(vals) == 0:
            return dict(mean=np.nan, max=np.nan, std=np.nan, range=np.nan)
        return dict(mean=float(np.mean(vals)),
                    max=float(np.max(vals)),
                    std=float(np.std(vals)),
                    range=float(np.max(vals) - np.min(vals)))
    except:
        return dict(mean=np.nan, max=np.nan, std=np.nan, range=np.nan)

# Buffer segments by 500m for zonal extraction
segs_buf = segs.copy()
segs_buf['geometry'] = segs.geometry.buffer(500)

records = []
with rasterio.open(dem_path) as dem_src:
    # Write slope and roughness as temporary in-memory rasters
    import tempfile, os

    def write_temp(arr, profile, nodata=-9999):
        p = profile.copy()
        p.update(dtype=rasterio.float32, count=1, nodata=nodata)
        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', **p) as dst:
            dst.write(arr.astype(np.float32), 1)
        return tmp.name

    slope_path    = write_temp(np.array(slope_arr),    profile)
    curve_path    = write_temp(np.array(curve_arr),    profile)
    rough_path    = write_temp(roughness_arr,           profile)

    with rasterio.open(slope_path) as sl_src, \
         rasterio.open(curve_path) as cu_src, \
         rasterio.open(rough_path) as ro_src:

        for idx, row in segs_buf.iterrows():
            geom = row.geometry.__geo_interface__
            elev  = zonal_stats(geom, dem_src, None)
            slope = zonal_stats(geom, sl_src,  None)
            curve = zonal_stats(geom, cu_src,  None)
            rough = zonal_stats(geom, ro_src,  None)

            records.append({
                'segment_id':        row['segment_id'],
                'road_name':         row['road_name'],
                'mean_elevation_m':  round(elev['mean'],  1),
                'elev_range_m':      round(elev['range'], 1),
                'mean_slope_deg':    round(slope['mean'], 2),
                'max_slope_deg':     round(slope['max'],  2),
                'std_slope_deg':     round(slope['std'],  2),
                'mean_curvature':    round(curve['mean'], 4),
                'terrain_roughness': round(rough['mean'], 2),
            })

            if (idx+1) % 50 == 0:
                print(f"  Processed {idx+1}/{len(segs)} segments...")

    # Cleanup temp files
    for p in [slope_path, curve_path, rough_path]:
        os.unlink(p)

# ── 4. Save feature table ─────────────────────────────────────────────────────
df = pd.DataFrame(records)
out = 'data/processed/terrain_features/terrain_features.csv'
df.to_csv(out, index=False)

print(f"\nTerrain features saved: {out}")
print(f"Shape: {df.shape}")
print("\nSample output:")
print(df.head(5).to_string(index=False))
print("\nPhase 3 (terrain features) complete.")
