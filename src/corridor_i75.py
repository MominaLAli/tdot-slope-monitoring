import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, MultiLineString
import warnings
warnings.filterwarnings('ignore')

print("Building I-75 Jellico corridor...")

# I-75 through Campbell County TN — steep mountain section near Jellico
# Known for rockfall and slope issues — documented in TDOT records
i75_coords = [
    (-84.15, 36.58), (-84.13, 36.52), (-84.10, 36.48),
    (-84.08, 36.44), (-84.05, 36.40), (-84.03, 36.36),
    (-84.00, 36.32), (-83.98, 36.28), (-83.95, 36.24),
    (-83.93, 36.20), (-83.90, 36.16), (-83.88, 36.12),
    (-83.85, 36.08), (-83.83, 36.04), (-83.80, 36.00),
]

corridor_line = LineString(i75_coords)
gdf_raw  = gpd.GeoDataFrame(geometry=[corridor_line], crs='EPSG:4326')
gdf_utm  = gdf_raw.to_crs(epsg=32617)
corridor = gdf_utm.geometry.iloc[0]
print(f"  I-75 corridor length: {corridor.length/1000:.1f} km")

# Buffer
buffer = gdf_utm.copy()
buffer['geometry'] = gdf_utm.geometry.buffer(1000)
buffer.to_file('data/processed/corridor/i75_study_buffer.geojson', driver='GeoJSON')

# Segments
def split_line(line, seg_m=804.7):
    segs, d = [], 0
    while d < line.length:
        segs.append(LineString([
            line.interpolate(d),
            line.interpolate(min(d+seg_m, line.length))
        ]))
        d += seg_m
    return segs

all_segs = split_line(corridor)
gdf_segs = gpd.GeoDataFrame(
    {'segment_id': range(1, len(all_segs)+1),
     'road_name':  'I-75',
     'length_m':   [round(s.length,1) for s in all_segs]},
    geometry=all_segs, crs='EPSG:32617'
)
gdf_segs.to_crs(epsg=4326).to_file(
    'data/processed/corridor/i75_segments.geojson', driver='GeoJSON')
gdf_segs.to_file('data/raw/roads/i75_corridor_utm.geojson', driver='GeoJSON')

print(f"  Segments: {len(gdf_segs)}")
print(f"  Saved: i75_segments.geojson")

# ── Terrain features for I-75 ─────────────────────────────────────────────────
print("\nDownloading DEM for I-75 corridor...")
import requests, rasterio
from rasterio.merge import merge
from rasterio.mask import mask

bounds = buffer.to_crs(epsg=4326).total_bounds
print(f"  Bounds: {bounds.round(4)}")

def download_dem_tile(bbox, out_path):
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

minx,miny,maxx,maxy = bounds
tile_w = (maxx-minx)/2
tiles  = []
for i in range(2):
    tx0 = minx + i*tile_w
    out = f'data/raw/dem_lidar/i75_dem_tile_{i+1}.tif'
    if download_dem_tile((tx0, miny, tx0+tile_w+0.05, maxy), out):
        tiles.append(out)
        print(f"  Tile {i+1}/2 downloaded")

# Merge
srcs   = [rasterio.open(t) for t in tiles]
mosaic, transform = merge(srcs)
meta   = srcs[0].meta.copy()
meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
for s in srcs: s.close()

merged = 'data/raw/dem_lidar/i75_dem_merged.tif'
with rasterio.open(merged, 'w', **meta) as dst:
    dst.write(mosaic)

# Clip
buf_utm = buffer.to_crs(epsg=32617)
with rasterio.open(merged) as src:
    clipped, clip_transform = mask(src, buf_utm.geometry, crop=True)
    clip_meta = src.meta.copy()
clip_meta.update(height=clipped.shape[1], width=clipped.shape[2],
                 transform=clip_transform)
clipped_path = 'data/raw/dem_lidar/i75_dem_clipped.tif'
with rasterio.open(clipped_path, 'w', **clip_meta) as dst:
    dst.write(clipped)

valid = clipped[0][clipped[0] > -9999]
print(f"  I-75 elevation range: {valid.min():.0f}m – {valid.max():.0f}m")
print("\nI-75 corridor setup complete.")
