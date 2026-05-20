import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Load data ─────────────────────────────────────────────────────────────────
df  = pd.read_csv('data/processed/model_table/combined_corridors.csv')
ts  = pd.read_csv('data/processed/insar_features/insar_timeseries.csv')
i40 = gpd.read_file('data/processed/corridor/segments.geojson')
i75 = gpd.read_file('data/processed/corridor/i75_segments.geojson')
i40['segment_id'] = i40['segment_id'].astype(int)
i75['segment_id'] = i75['segment_id'].astype(int) + 1000
i40['corridor']   = 'I-40'
i75['corridor']   = 'I-75'
gdf = pd.concat([i40, i75], ignore_index=True)
gdf = gdf.merge(
    df[['segment_id','risk_score','risk_category',
        'mean_slope_deg','mean_disp_mm','rock_class',
        'AADT_2022','mean_coherence','corridor']],
    on='segment_id', how='left'
)

COLORS  = {'Low':'#26de81','Moderate':'#4a90d9','High':'#ff9f43','Critical':'#ff4b4b'}
DARK_BG = '#0f1117'
CARD_BG = '#1e2235'
ACCENT  = '#64ffda'
TEXT    = '#e0e0e0'
MUTED   = '#8892b0'
fig_dir = Path('outputs/figures')
map_dir = Path('outputs/maps')
plt.style.use('dark_background')
print("Generating figures...")

# ── Figure 1: Risk Map ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16,7), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
for cat, color in COLORS.items():
    sub = gdf[gdf['risk_category']==cat]
    if len(sub)==0: continue
    lw = {'Low':1.5,'Moderate':2.5,'High':3.5,'Critical':5.0}[cat]
    sub.plot(ax=ax, color=color, linewidth=lw, label=cat)
patches = [mpatches.Patch(color=c,label=l) for l,c in COLORS.items()]
ax.legend(handles=patches, loc='lower right', facecolor=CARD_BG,
          edgecolor=MUTED, labelcolor=TEXT, fontsize=11,
          title='Risk Level', title_fontsize=12)
ax.set_title('TDOT Slope Risk — I-40 + I-75 Corridors, East Tennessee',
             color=TEXT, fontsize=15, fontweight='bold', pad=15)
ax.text(0.01,0.97,'I-40: Cookeville→NC Border  |  I-75: Jellico Mountain Section',
        transform=ax.transAxes, color=MUTED, fontsize=9, va='top')
ax.text(0.01,0.92,'Data: Sentinel-1 InSAR (ASF HyP3) + USGS 3DEP + USGS Geology + FHWA HPMS',
        transform=ax.transAxes, color=MUTED, fontsize=8, va='top')
ax.set_xlabel('Longitude', color=MUTED); ax.set_ylabel('Latitude', color=MUTED)
ax.tick_params(colors=MUTED)
for s in ax.spines.values(): s.set_edgecolor(MUTED)
plt.tight_layout()
plt.savefig(fig_dir/'fig1_risk_map.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  fig1_risk_map.png")

# ── Figure 2: Risk Score Profiles ────────────────────────────────────────────
fig, axes = plt.subplots(2,1, figsize=(14,8), facecolor=DARK_BG)
for ax, corridor, color in zip(axes,['I-40','I-75'],[ACCENT,'#ff9f43']):
    sub = df[df['corridor']==corridor].sort_values('segment_id').reset_index(drop=True)
    ax.set_facecolor(DARK_BG)
    for cat, c in COLORS.items():
        s = sub[sub['risk_category']==cat]
        ax.scatter(s.index, s['risk_score'], color=c, s=18, zorder=3, label=cat)
    ax.plot(sub.index, sub['risk_score'], color=color, linewidth=0.8, alpha=0.5, zorder=2)
    ax.axhline(0.50, color='#8892b0', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(0.75, color='#ff4b4b', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(f'{corridor} — Risk Score Profile', color=TEXT, fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk Score', color=MUTED)
    ax.set_xlabel('Segment Position', color=MUTED)
    ax.tick_params(colors=MUTED); ax.set_ylim(0,1)
    ax.grid(color='#2d3147', alpha=0.5)
    for s in ax.spines.values(): s.set_edgecolor('#2d3147')
    if corridor=='I-40':
        ax.legend(loc='upper left', facecolor=CARD_BG, edgecolor=MUTED,
                  labelcolor=TEXT, fontsize=8, ncol=3)
fig.suptitle('Risk Score Along I-40 and I-75 Corridors',
             color=TEXT, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir/'fig2_risk_profiles.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  fig2_risk_profiles.png")

# ── Figure 3: Feature Importance ─────────────────────────────────────────────
fi = pd.read_csv('outputs/tables/feature_importance.csv',
                 names=['feature','importance'], header=0)\
       .sort_values('importance', ascending=True).tail(12)
fig, ax = plt.subplots(figsize=(10,7), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
colors_fi = plt.cm.YlOrRd(np.linspace(0.3,1.0,len(fi)))
bars = ax.barh(fi['feature'], fi['importance'], color=colors_fi, edgecolor='none')
for bar, val in zip(bars, fi['importance']):
    ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', color=TEXT, fontsize=9)
ax.set_xlabel('Feature Importance', color=MUTED, fontsize=11)
ax.set_title('Random Forest Feature Importance\nI-40 + I-75 Combined (432 segments)',
             color=TEXT, fontsize=13, fontweight='bold')
ax.tick_params(colors=MUTED)
ax.grid(axis='x', color='#2d3147', alpha=0.5)
for s in ax.spines.values(): s.set_edgecolor('#2d3147')
plt.tight_layout()
plt.savefig(fig_dir/'fig3_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  fig3_feature_importance.png")

# ── Figure 4: Displacement Time Series ───────────────────────────────────────
top5    = [149, 92, 138, 91, 139]
palette = [ACCENT,'#ff9f43','#4a90d9','#ff4b4b','#26de81']
fig, ax = plt.subplots(figsize=(13,6), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
for sid, color in zip(top5, palette):
    sub = ts[ts['segment_id']==sid].dropna(subset=['disp_mm']).sort_values('date')
    if len(sub)==0: continue
    cat = df[df['segment_id']==sid]['risk_category'].values
    ax.plot(pd.to_datetime(sub['date']), sub['disp_mm'],
            color=color, linewidth=2, marker='o', markersize=3,
            label=f'Seg {sid} [{cat[0] if len(cat)>0 else "?"}]')
ax.axhline(0, color=MUTED, linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Date', color=MUTED, fontsize=11)
ax.set_ylabel('Vertical Displacement (mm)', color=MUTED, fontsize=11)
ax.set_title('InSAR Displacement Time Series — Top 5 Most Active Segments\n'
             'Sentinel-1, ASF HyP3, 35 pairs, Jan 2022 – Jan 2024',
             color=TEXT, fontsize=13, fontweight='bold')
ax.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=10)
ax.tick_params(colors=MUTED)
ax.grid(color='#2d3147', alpha=0.5)
for s in ax.spines.values(): s.set_edgecolor('#2d3147')
plt.tight_layout()
plt.savefig(fig_dir/'fig4_displacement_timeseries.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  fig4_displacement_timeseries.png")

# ── Figure 5: Risk by Rock Class ─────────────────────────────────────────────
rock_agg = df.groupby('rock_class').agg(
    mean_risk=('risk_score','mean'),
    count=('segment_id','count')
).reset_index().sort_values('mean_risk', ascending=False)
rock_colors = ['#ff4b4b','#ff9f43','#4a90d9','#26de81','#8892b0']
fig, axes = plt.subplots(1,2, figsize=(14,6), facecolor=DARK_BG)
for ax in axes: ax.set_facecolor(DARK_BG)
bars = axes[0].bar(rock_agg['rock_class'], rock_agg['mean_risk'],
                   color=rock_colors[:len(rock_agg)], edgecolor='none')
for bar, val in zip(bars, rock_agg['mean_risk']):
    axes[0].text(bar.get_x()+bar.get_width()/2, val+0.005,
                 f'{val:.3f}', ha='center', color=TEXT, fontsize=9)
axes[0].set_title('Mean Risk Score by Rock Class', color=TEXT, fontsize=12, fontweight='bold')
axes[0].set_ylabel('Mean Risk Score', color=MUTED)
axes[0].tick_params(colors=MUTED, axis='both')
axes[0].tick_params(axis='x', rotation=20)
axes[0].grid(axis='y', color='#2d3147', alpha=0.5)
for s in axes[0].spines.values(): s.set_edgecolor('#2d3147')
bars2 = axes[1].bar(rock_agg['rock_class'], rock_agg['count'],
                    color=rock_colors[:len(rock_agg)], edgecolor='none')
for bar, val in zip(bars2, rock_agg['count']):
    axes[1].text(bar.get_x()+bar.get_width()/2, val+1,
                 f'n={val}', ha='center', color=TEXT, fontsize=9)
axes[1].set_title('Segment Count by Rock Class', color=TEXT, fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Segments', color=MUTED)
axes[1].tick_params(colors=MUTED, axis='both')
axes[1].tick_params(axis='x', rotation=20)
axes[1].grid(axis='y', color='#2d3147', alpha=0.5)
for s in axes[1].spines.values(): s.set_edgecolor('#2d3147')
fig.suptitle('Slope Risk by Geology — USGS Tennessee State Geology Map',
             color=TEXT, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir/'fig5_risk_by_geology.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  fig5_risk_by_geology.png")

# ── Figure 6: Coherence vs Displacement ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(11,7), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
for cat, color in COLORS.items():
    sub = df[df['risk_category']==cat]
    if len(sub)==0: continue
    for corr, mk in [('I-40','o'),('I-75','D')]:
        s = sub[sub['corridor']==corr]
        if len(s)==0: continue
        ax.scatter(s['mean_coherence'], s['mean_disp_mm'],
                   color=color, marker=mk, s=35, alpha=0.8,
                   label=f'{cat} ({corr})' if corr=='I-40' else '_nolegend_')
ax.axvline(0.3, color='#ff4b4b', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Low coherence threshold')
ax.axhline(0,   color=MUTED,    linestyle='--', linewidth=1,   alpha=0.5)
ax.set_xlabel('Mean InSAR Coherence', color=MUTED, fontsize=11)
ax.set_ylabel('Mean Displacement (mm)', color=MUTED, fontsize=11)
ax.set_title('InSAR Coherence vs Displacement by Risk Category\n'
             'Circles=I-40 · Diamonds=I-75',
             color=TEXT, fontsize=13, fontweight='bold')
ax.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=9, ncol=2)
ax.tick_params(colors=MUTED)
ax.grid(color='#2d3147', alpha=0.5)
for s in ax.spines.values(): s.set_edgecolor('#2d3147')
plt.tight_layout()
plt.savefig(fig_dir/'fig6_coherence_vs_displacement.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  fig6_coherence_vs_displacement.png")

# ── Figure 7: Risk Distribution Summary ──────────────────────────────────────
fig, axes = plt.subplots(1,3, figsize=(15,5), facecolor=DARK_BG)
for ax in axes: ax.set_facecolor(DARK_BG)

cat_counts = df['risk_category'].value_counts()\
               .reindex(['Low','Moderate','High','Critical'], fill_value=0)
wedges, texts, autotexts = axes[0].pie(
    cat_counts.values, labels=cat_counts.index,
    colors=[COLORS[c] for c in cat_counts.index],
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor=DARK_BG, linewidth=2))
for t in texts:     t.set_color(TEXT)
for t in autotexts: t.set_color(DARK_BG); t.set_fontweight('bold')
axes[0].set_title('Overall Risk Distribution\n(432 segments)',
                  color=TEXT, fontsize=11, fontweight='bold')

x = np.arange(4)
cats = ['Low','Moderate','High','Critical']
for i, (corr, color) in enumerate([('I-40',ACCENT),('I-75','#ff9f43')]):
    sub  = df[df['corridor']==corr]
    vals = sub['risk_category'].value_counts().reindex(cats, fill_value=0)
    axes[1].bar(x+i*0.35, vals.values, 0.35, label=corr,
                color=[COLORS[c] for c in cats],
                edgecolor=DARK_BG, linewidth=1.5)
axes[1].set_xticks(x+0.175); axes[1].set_xticklabels(cats, color=MUTED)
axes[1].set_title('Risk by Corridor', color=TEXT, fontsize=11, fontweight='bold')
axes[1].set_ylabel('Segments', color=MUTED)
axes[1].tick_params(colors=MUTED)
axes[1].grid(axis='y', color='#2d3147', alpha=0.5)
for s in axes[1].spines.values(): s.set_edgecolor('#2d3147')
axes[1].legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)

axes[2].hist(df[df['corridor']=='I-40']['risk_score'],
             bins=30, color=ACCENT,    alpha=0.7, label='I-40', edgecolor='none')
axes[2].hist(df[df['corridor']=='I-75']['risk_score'],
             bins=20, color='#ff9f43', alpha=0.7, label='I-75', edgecolor='none')
axes[2].axvline(0.50, color='#ff9f43', linestyle='--', linewidth=1.5, alpha=0.8)
axes[2].axvline(0.75, color='#ff4b4b', linestyle='--', linewidth=1.5, alpha=0.8)
axes[2].set_xlabel('Risk Score',   color=MUTED)
axes[2].set_ylabel('Segments',     color=MUTED)
axes[2].set_title('Risk Score Distribution', color=TEXT, fontsize=11, fontweight='bold')
axes[2].tick_params(colors=MUTED)
axes[2].grid(color='#2d3147', alpha=0.5)
for s in axes[2].spines.values(): s.set_edgecolor('#2d3147')
axes[2].legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)

fig.suptitle('TDOT Slope Risk — Summary Statistics',
             color=TEXT, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir/'fig7_risk_summary.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  fig7_risk_summary.png")

# ── Save outputs/maps ─────────────────────────────────────────────────────────
gdf_out = gdf[['segment_id','corridor','geometry','risk_score',
               'risk_category','mean_slope_deg','mean_disp_mm',
               'rock_class','AADT_2022']].copy()
gdf_out.to_crs(epsg=4326).to_file('outputs/maps/risk_segments.geojson', driver='GeoJSON')
print("  outputs/maps/risk_segments.geojson")

df.nlargest(10,'risk_score')[
    ['segment_id','corridor','risk_score','risk_category',
     'mean_slope_deg','elev_range_m','mean_disp_mm',
     'mean_coherence','rock_class','AADT_2022']
].round(3).to_csv('outputs/tables/top_risk_segments.csv', index=False)
print("  outputs/tables/top_risk_segments.csv")

print(f"\nAll outputs generated:")
print(f"  Figures: {len(list(fig_dir.glob('*.png')))} PNG files in outputs/figures/")
print(f"  Maps:    {len(list(map_dir.glob('*.geojson')))} GeoJSON in outputs/maps/")
print(f"  Tables:  {len(list(Path('outputs/tables').glob('*.csv')))} CSV in outputs/tables/")
print("\nDone.")
