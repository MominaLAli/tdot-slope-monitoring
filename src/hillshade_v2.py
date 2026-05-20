import numpy as np
import rasterio
from rasterio.enums import Resampling
import requests
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DARK_BG = '#0f1117'
TEXT    = '#e0e0e0'
MUTED   = '#8892b0'

def compute_hillshade(dem, azimuth=315, altitude=45):
    az  = np.radians(360 - azimuth + 90)
    alt = np.radians(altitude)
    dy, dx = np.gradient(dem)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    hs = (np.sin(alt)*np.cos(slope) +
          np.cos(alt)*np.sin(slope)*np.cos(az - aspect))
    return np.clip(hs, 0, 1)

def download_wide_dem(bounds_wgs84, out_path, size=1024):
    """Download DEM with wide buffer around corridor."""
    minx, miny, maxx, maxy = bounds_wgs84
    # Add 0.15 degree buffer (~15km) on each side
    buf = 0.15
    url = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/"
        "3DEPElevation/ImageServer/exportImage"
        f"?bbox={minx-buf},{miny-buf},{maxx+buf},{maxy+buf}"
        f"&bboxSR=4326&size={size},{size}&imageSR=4326"
        "&format=tiff&pixelType=F32&noDataInterpretation=esriNoDataMatchAny"
        "&interpolation=RSP_BilinearInterpolation&f=image"
    )
    r = requests.get(url, timeout=180)
    if r.status_code == 200 and len(r.content) > 10000:
        with open(out_path, 'wb') as f:
            f.write(r.content)
        return True
    return False

# ── Download wide DEMs ────────────────────────────────────────────────────────
print("Downloading wide-area DEMs for hillshade...")

# I-40 bounds (WGS84)
i40_bounds = (-85.51, 35.74, -82.49, 36.16)
i75_bounds = (-84.16, 35.99, -83.79, 36.59)

p_i40 = Path('data/raw/dem_lidar/i40_wide_dem.tif')
p_i75 = Path('data/raw/dem_lidar/i75_wide_dem.tif')

if not p_i40.exists():
    ok = download_wide_dem(i40_bounds, p_i40, size=1024)
    print(f"  I-40 wide DEM: {'OK' if ok else 'FAILED'}")
else:
    print("  I-40 wide DEM: already exists")

if not p_i75.exists():
    ok = download_wide_dem(i75_bounds, p_i75, size=1024)
    print(f"  I-75 wide DEM: {'OK' if ok else 'FAILED'}")
else:
    print("  I-75 wide DEM: already exists")

# ── Load corridor lines in WGS84 ──────────────────────────────────────────────
i40_line = gpd.read_file('data/processed/corridor/segments.geojson').to_crs(epsg=4326)
i75_line = gpd.read_file('data/processed/corridor/i75_segments.geojson').to_crs(epsg=4326)

def make_hillshade_figure(dem_path, corridor_gdf, title, subtitle,
                           highway_color, out_path):
    with rasterio.open(dem_path) as src:
        dem    = src.read(1).astype(float)
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    dem[dem < -9000] = np.nan
    valid = dem[~np.isnan(dem)]
    print(f"  {title[:20]}... elev: {valid.min():.0f}–{valid.max():.0f}m")

    hs       = compute_hillshade(np.where(np.isnan(dem), 0, dem))
    norm_dem = (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem))
    colored  = plt.cm.terrain(norm_dem)
    blended  = colored.copy()
    for c in range(3):
        blended[:,:,c] = colored[:,:,c] * (0.55 + 0.45 * hs)
    blended = np.clip(blended, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK_BG)
    fig.suptitle(f'{title}\n{subtitle}',
                 color=TEXT, fontsize=14, fontweight='bold')

    # Left: pure hillshade
    axes[0].set_facecolor(DARK_BG)
    axes[0].imshow(hs, cmap='gray', extent=extent,
                   origin='upper', interpolation='bilinear', aspect='auto')
    corridor_gdf.plot(ax=axes[0], color=highway_color,
                      linewidth=2, label='Highway')
    axes[0].set_title('Terrain Hillshade', color=TEXT, fontsize=12)
    axes[0].set_xlabel('Longitude', color=MUTED, fontsize=9)
    axes[0].set_ylabel('Latitude',  color=MUTED, fontsize=9)
    axes[0].tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(facecolor='#1e2235', edgecolor='#2d3147',
                   labelcolor=TEXT, fontsize=9)
    for s in axes[0].spines.values(): s.set_edgecolor('#2d3147')
    axes[0].text(0.02, 0.97,
                 f"Min: {valid.min():.0f}m\nMax: {valid.max():.0f}m\n"
                 f"Range: {valid.max()-valid.min():.0f}m",
                 transform=axes[0].transAxes, color=TEXT, fontsize=9,
                 va='top', bbox=dict(boxstyle='round', facecolor='#1e2235',
                                     edgecolor='#2d3147', alpha=0.85))

    # Right: elevation + hillshade blend
    axes[1].set_facecolor(DARK_BG)
    axes[1].imshow(blended, extent=extent, origin='upper',
                   interpolation='bilinear', aspect='auto')
    corridor_gdf.plot(ax=axes[1], color='#ff4b4b',
                      linewidth=2.5, label='Highway')
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.terrain,
        norm=plt.Normalize(vmin=int(valid.min()), vmax=int(valid.max())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], shrink=0.75, pad=0.02)
    cbar.set_label('Elevation (m)', color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    axes[1].set_title('Elevation + Hillshade Blend', color=TEXT, fontsize=12)
    axes[1].set_xlabel('Longitude', color=MUTED, fontsize=9)
    axes[1].set_ylabel('Latitude',  color=MUTED, fontsize=9)
    axes[1].tick_params(colors=MUTED, labelsize=8)
    axes[1].legend(facecolor='#1e2235', edgecolor='#2d3147',
                   labelcolor=TEXT, fontsize=9)
    for s in axes[1].spines.values(): s.set_edgecolor('#2d3147')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out_path}")

# ── Generate figures ──────────────────────────────────────────────────────────
print("\nGenerating hillshade figures...")

make_hillshade_figure(
    dem_path      = p_i40,
    corridor_gdf  = i40_line,
    title         = 'I-40 Mountain Corridor — East Tennessee',
    subtitle      = 'Cookeville → NC Border · 274 km · USGS 3DEP 30m DEM',
    highway_color = '#64ffda',
    out_path      = 'outputs/figures/fig8_i40_hillshade.png'
)

make_hillshade_figure(
    dem_path      = p_i75,
    corridor_gdf  = i75_line,
    title         = 'I-75 Jellico Mountain Section — Campbell County TN',
    subtitle      = 'Known Rockfall and Slope Hazard Area · 72 km · USGS 3DEP 30m DEM',
    highway_color = '#ff9f43',
    out_path      = 'outputs/figures/fig9_i75_hillshade.png'
)

# ── Combined comparison ───────────────────────────────────────────────────────
print("\nGenerating combined terrain comparison...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=DARK_BG)
fig.suptitle('East Tennessee Highway Corridor Terrain — USGS 3DEP 30m DEM',
             color=TEXT, fontsize=15, fontweight='bold')

for ax, dem_path, gdf_line, label, hw_color in zip(
    axes,
    [p_i40, p_i75],
    [i40_line, i75_line],
    ['I-40 · Cookeville → NC Border (274 km)',
     'I-75 · Jellico Mountain Section (72 km)'],
    ['#64ffda','#ff9f43']
):
    with rasterio.open(dem_path) as src:
        dem    = src.read(1).astype(float)
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
    dem[dem < -9000] = np.nan
    valid    = dem[~np.isnan(dem)]
    hs       = compute_hillshade(np.where(np.isnan(dem), 0, dem))
    norm_dem = (dem-np.nanmin(dem))/(np.nanmax(dem)-np.nanmin(dem))
    colored  = plt.cm.terrain(norm_dem)
    blended  = colored.copy()
    for c in range(3):
        blended[:,:,c] = colored[:,:,c]*(0.55 + 0.45*hs)
    blended = np.clip(blended, 0, 1)

    ax.set_facecolor(DARK_BG)
    ax.imshow(blended, extent=extent, origin='upper',
              interpolation='bilinear', aspect='auto')
    gdf_line.plot(ax=ax, color=hw_color, linewidth=2.5, label='Highway')
    ax.text(0.02, 0.97,
            f"Min: {valid.min():.0f}m\nMax: {valid.max():.0f}m\n"
            f"Range: {valid.max()-valid.min():.0f}m",
            transform=ax.transAxes, color=TEXT, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='#1e2235',
                      edgecolor='#2d3147', alpha=0.85))
    ax.set_title(label, color=TEXT, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude', color=MUTED, fontsize=9)
    ax.set_ylabel('Latitude',  color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(facecolor='#1e2235', edgecolor='#2d3147',
              labelcolor=TEXT, fontsize=9)
    for s in ax.spines.values(): s.set_edgecolor('#2d3147')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.terrain,
        norm=plt.Normalize(vmin=int(valid.min()),vmax=int(valid.max())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Elevation (m)', color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)

plt.tight_layout()
plt.savefig('outputs/figures/fig10_terrain_comparison.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  Saved: outputs/figures/fig10_terrain_comparison.png")
print("\nHillshade v2 complete.")
