import requests
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import os, math
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load corridor buffer (defines download area) ──────────────────────────
buffer = gpd.read_file('data/processed/corridor/study_buffer.geojson')
bounds = buffer.to_crs(epsg=4326).total_bounds  # (minx, miny, maxx, maxy)
print(f"  Corridor bounds (WGS84): {bounds.round(4)}")

# ── 2. Download DEM tiles from USGS 3DEP (1 arc-second ~30m resolution) ──────
# Split into tiles to avoid timeout
def download_dem_tile(bbox, out_path):
    """Download a single DEM tile via USGS TNM API."""
    minx, miny, maxx, maxy = bbox
    url = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/"
        "3DEPElevation/ImageServer/exportImage"
        f"?bbox={minx},{miny},{maxx},{maxy}"
        "&bboxSR=4326&size=512,512&imageSR=32617"
        "&format=tiff&pixelType=F32&noDataInterpretation=esriNoDataMatchAny"
        "&interpolation=RSP_BilinearInterpolation&f=image"
    )
    r = requests.get(url, timeout=120)
    if r.status_code == 200 and len(r.content) > 1000:
        with open(out_path, 'wb') as f:
            f.write(r.content)
        return True
    return False

# Split corridor into 4 tiles east→west to keep requests small
minx, miny, maxx, maxy = bounds
tile_width = (maxx - minx) / 4
tiles = []

print("Downloading DEM tiles from USGS 3DEP...")
for i in range(4):
    tx0 = minx + i * tile_width
    tx1 = tx0 + tile_width + 0.05  # slight overlap
    out = f'data/raw/dem_lidar/dem_tile_{i+1}.tif'
    ok = download_dem_tile((tx0, miny, tx1, maxy), out)
    if ok:
        tiles.append(out)
        print(f"  Tile {i+1}/4 downloaded: {out}")
    else:
        print(f"  Tile {i+1}/4 FAILED — skipping")

if not tiles:
    print("ERROR: No tiles downloaded. Check internet connection.")
    exit(1)

# ── 3. Merge tiles into one DEM ───────────────────────────────────────────────
print("Merging tiles...")
src_files = [rasterio.open(t) for t in tiles]
mosaic, out_transform = merge(src_files)
out_meta = src_files[0].meta.copy()
out_meta.update({"driver":"GTiff","height":mosaic.shape[1],
                  "width":mosaic.shape[2],"transform":out_transform})
for s in src_files:
    s.close()

merged_path = 'data/raw/dem_lidar/dem_merged.tif'
with rasterio.open(merged_path, 'w', **out_meta) as dest:
    dest.write(mosaic)
print(f"  Merged DEM saved: {merged_path}")

# ── 4. Clip to corridor buffer ────────────────────────────────────────────────
buffer_utm = buffer.to_crs(epsg=32617)
with rasterio.open(merged_path) as src:
    clipped, clipped_transform = mask(src, buffer_utm.geometry, crop=True)
    clipped_meta = src.meta.copy()

clipped_meta.update({"height": clipped.shape[1], "width": clipped.shape[2],
                      "transform": clipped_transform})
clipped_path = 'data/raw/dem_lidar/dem_clipped.tif'
with rasterio.open(clipped_path, 'w', **clipped_meta) as dest:
    dest.write(clipped)

# Quick stats
valid = clipped[0][clipped[0] > -9999]
print(f"  Clipped DEM saved: {clipped_path}")
print(f"  Elevation range: {valid.min():.0f}m – {valid.max():.0f}m")
print(f"  Shape: {clipped.shape[1]} x {clipped.shape[2]} pixels")
print("\nPhase 2 (DEM download) complete.")
