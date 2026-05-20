import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import warnings
warnings.filterwarnings('ignore')

print("Building I-40 corridor from hardcoded coordinates...")

# I-40 mountain corridor: Cookeville TN → NC border
# Coordinates sampled along the actual highway (lon, lat)
i40_coords = [
    (-85.50, 36.15), (-85.30, 36.12), (-85.00, 36.08),
    (-84.80, 36.05), (-84.50, 35.98), (-84.20, 35.95),
    (-83.90, 35.92), (-83.70, 35.88), (-83.50, 35.85),
    (-83.20, 35.82), (-83.00, 35.79), (-82.90, 35.78),
    (-82.80, 35.77), (-82.60, 35.76), (-82.50, 35.75),
]

corridor_line = LineString(i40_coords)
gdf_raw = gpd.GeoDataFrame(geometry=[corridor_line], crs='EPSG:4326')
gdf_utm  = gdf_raw.to_crs(epsg=32617)
corridor = gdf_utm.geometry.iloc[0]
print(f"  Corridor length: {corridor.length/1000:.1f} km")

# 1 km buffer
buffer = gdf_utm.copy()
buffer['geometry'] = gdf_utm.geometry.buffer(1000)
buffer.to_file('data/processed/corridor/study_buffer.geojson', driver='GeoJSON')
print("  Saved: study_buffer.geojson")

# Split into 0.5-mile (804.7m) segments
def split_line(line, seg_m=804.7):
    segs = []
    d = 0
    while d < line.length:
        segs.append(LineString([
            line.interpolate(d),
            line.interpolate(min(d + seg_m, line.length))
        ]))
        d += seg_m
    return segs

all_segs = []
if isinstance(corridor, MultiLineString):
    for part in corridor.geoms:
        all_segs.extend(split_line(part))
else:
    all_segs = split_line(corridor)

gdf_segs = gpd.GeoDataFrame(
    {'segment_id': range(1, len(all_segs)+1),
     'road_name':  'I-40',
     'length_m':   [round(s.length, 1) for s in all_segs]},
    geometry=all_segs, crs='EPSG:32617'
)

gdf_segs.to_crs(epsg=4326).to_file('data/processed/corridor/segments.geojson', driver='GeoJSON')
gdf_segs.to_file('data/raw/roads/i40_corridor_utm.geojson', driver='GeoJSON')

print(f"  Segments created: {len(gdf_segs)}")
print(f"\nSegment preview:")
print(gdf_segs[['segment_id','road_name','length_m']].head(5).to_string(index=False))
print("\nPhase 1 complete.")
