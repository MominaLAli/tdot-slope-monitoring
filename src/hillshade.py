import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("Generating hillshade terrain visualizations...")

def compute_hillshade(dem, azimuth=315, altitude=45):
    """Compute hillshade from DEM array."""
    az  = np.radians(360 - azimuth + 90)
    alt = np.radians(altitude)

    # Compute gradients
    dy, dx = np.gradient(dem)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)

    # Hillshade formula
    hs = (np.sin(alt) * np.cos(slope) +
          np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = np.clip(hs, 0, 1)
    return hs

def make_terrain_figure(dem_path, title, out_path, corridor_gdf=None):
    with rasterio.open(dem_path) as src:
        # Resample to manageable size if too large
        scale = 1
        w, h  = src.width, src.height
        if w * h > 2_000_000:
            scale = 2
        dem = src.read(
            1,
            out_shape=(1, h//scale, w//scale),
            resampling=Resampling.bilinear
        ).astype(float)
        dem[dem < -9000] = np.nan
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    hs = compute_hillshade(np.where(np.isnan(dem), 0, dem))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              facecolor='#0f1117')
    fig.suptitle(title, color='#e0e0e0', fontsize=14,
                 fontweight='bold', y=1.01)

    # ── Left: Hillshade only ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#0f1117')
    ax1.imshow(hs, cmap='gray', extent=extent,
               origin='upper', interpolation='bilinear')
    if corridor_gdf is not None:
        corridor_gdf.plot(ax=ax1, color='#64ffda',
                          linewidth=1.5, label='Highway')
        ax1.legend(facecolor='#1e2235', edgecolor='#2d3147',
                   labelcolor='#e0e0e0', fontsize=9)
    ax1.set_title('Terrain Hillshade (USGS 3DEP 30m)',
                  color='#e0e0e0', fontsize=11)
    ax1.set_xlabel('Easting (m)',  color='#8892b0', fontsize=8)
    ax1.set_ylabel('Northing (m)', color='#8892b0', fontsize=8)
    ax1.tick_params(colors='#8892b0', labelsize=7)
    for s in ax1.spines.values(): s.set_edgecolor('#2d3147')

    # ── Right: Elevation colored + hillshade blend ───────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#0f1117')
    norm_dem = (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem))
    cmap_terrain = plt.cm.terrain
    colored = cmap_terrain(norm_dem)

    # Blend with hillshade
    blended = colored.copy()
    for c in range(3):
        blended[:,:,c] = colored[:,:,c] * (0.6 + 0.4 * hs)
    blended = np.clip(blended, 0, 1)

    im = ax2.imshow(blended, extent=extent,
                    origin='upper', interpolation='bilinear')
    if corridor_gdf is not None:
        corridor_gdf.plot(ax=ax2, color='#ff4b4b',
                          linewidth=2.0, label='Highway')
        ax2.legend(facecolor='#1e2235', edgecolor='#2d3147',
                   labelcolor='#e0e0e0', fontsize=9)

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap_terrain,
        norm=plt.Normalize(vmin=int(np.nanmin(dem)),
                           vmax=int(np.nanmax(dem))))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.7, pad=0.02)
    cbar.set_label('Elevation (m)', color='#e0e0e0', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#8892b0')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8892b0', fontsize=8)

    ax2.set_title('Elevation + Hillshade Blend',
                  color='#e0e0e0', fontsize=11)
    ax2.set_xlabel('Easting (m)',  color='#8892b0', fontsize=8)
    ax2.set_ylabel('Northing (m)', color='#8892b0', fontsize=8)
    ax2.tick_params(colors='#8892b0', labelsize=7)
    for s in ax2.spines.values(): s.set_edgecolor('#2d3147')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0f1117')
    plt.close()
    print(f"  Saved: {out_path}")

# ── Load corridor geometries ──────────────────────────────────────────────────
import geopandas as gpd

i40_gdf = gpd.read_file('data/raw/roads/i40_corridor_utm.geojson')
i75_gdf = gpd.read_file('data/raw/roads/i75_corridor_utm.geojson')

# ── Generate I-40 hillshade ───────────────────────────────────────────────────
make_terrain_figure(
    dem_path      = 'data/raw/dem_lidar/dem_clipped.tif',
    title         = 'I-40 Mountain Corridor — East Tennessee Terrain\n'
                    'Cookeville → NC Border · USGS 3DEP 30m DEM',
    out_path      = 'outputs/figures/fig8_i40_hillshade.png',
    corridor_gdf  = i40_gdf
)

# ── Generate I-75 hillshade ───────────────────────────────────────────────────
make_terrain_figure(
    dem_path      = 'data/raw/dem_lidar/i75_dem_clipped.tif',
    title         = 'I-75 Jellico Mountain Section — Campbell County Tennessee\n'
                    'USGS 3DEP 30m DEM · Known Rockfall and Slope Hazard Area',
    out_path      = 'outputs/figures/fig9_i75_hillshade.png',
    corridor_gdf  = i75_gdf
)

# ── Combined side-by-side ─────────────────────────────────────────────────────
print("  Generating combined comparison figure...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='#0f1117')
fig.suptitle('East Tennessee Highway Corridor Terrain — USGS 3DEP 30m DEM',
             color='#e0e0e0', fontsize=15, fontweight='bold')

for ax, dem_path, gdf_line, label, color in zip(
    axes,
    ['data/raw/dem_lidar/dem_clipped.tif',
     'data/raw/dem_lidar/i75_dem_clipped.tif'],
    [i40_gdf, i75_gdf],
    ['I-40 · Cookeville → NC Border (274 km)',
     'I-75 · Jellico Mountain Section (72 km)'],
    ['#64ffda','#ff9f43']
):
    with rasterio.open(dem_path) as src:
        w, h   = src.width, src.height
        scale  = 2 if w*h > 2_000_000 else 1
        dem    = src.read(1,
                          out_shape=(1,h//scale,w//scale),
                          resampling=Resampling.bilinear
                         ).astype(float)
        dem[dem < -9000] = np.nan
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    hs       = compute_hillshade(np.where(np.isnan(dem), 0, dem))
    norm_dem = (dem - np.nanmin(dem))/(np.nanmax(dem)-np.nanmin(dem))
    colored  = plt.cm.terrain(norm_dem)
    blended  = colored.copy()
    for c in range(3):
        blended[:,:,c] = colored[:,:,c] * (0.6 + 0.4*hs)
    blended = np.clip(blended, 0, 1)

    ax.set_facecolor('#0f1117')
    ax.imshow(blended, extent=extent, origin='upper',
              interpolation='bilinear')
    gdf_line.plot(ax=ax, color=color, linewidth=2.5, label='Highway')

    # Elevation stats annotation
    ax.text(0.02, 0.97,
            f"Min: {int(np.nanmin(dem))}m\n"
            f"Max: {int(np.nanmax(dem))}m\n"
            f"Range: {int(np.nanmax(dem)-np.nanmin(dem))}m",
            transform=ax.transAxes, color='#e0e0e0',
            fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='#1e2235',
                      edgecolor='#2d3147', alpha=0.8))

    ax.set_title(label, color='#e0e0e0', fontsize=12, fontweight='bold')
    ax.set_xlabel('Easting (m)',  color='#8892b0', fontsize=8)
    ax.set_ylabel('Northing (m)', color='#8892b0', fontsize=8)
    ax.tick_params(colors='#8892b0', labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor('#2d3147')
    ax.legend(facecolor='#1e2235', edgecolor='#2d3147',
              labelcolor='#e0e0e0', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/fig10_terrain_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved: outputs/figures/fig10_terrain_comparison.png")

print(f"\nTotal figures now: "
      f"{len(list(Path('outputs/figures').glob('*.png')))}")
print("\nHillshade generation complete.")
