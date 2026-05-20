import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

DARK_BG = '#0f1117'
CARD_BG = '#1e2235'
ACCENT  = '#64ffda'
TEXT    = '#e0e0e0'
MUTED   = '#8892b0'
ORANGE  = '#ff9f43'
RED     = '#ff4b4b'
BLUE    = '#4a90d9'
GREEN   = '#26de81'
plt.style.use('dark_background')

print("Loading data...")
ts   = pd.read_csv('data/processed/insar_features/insar_timeseries.csv')
df   = pd.read_csv('data/processed/model_table/full_feature_table.csv')
segs = gpd.read_file('data/processed/corridor/segments.geojson')

s149_ts = ts[ts['segment_id']==149].dropna(subset=['disp_mm'])\
            .sort_values('date').copy()
s149_ts['date_dt'] = pd.to_datetime(s149_ts['date'])
s149_ts['days']    = (s149_ts['date_dt'] - s149_ts['date_dt'].iloc[0]).dt.days

coeffs             = np.polyfit(s149_ts['days'], s149_ts['disp_mm'], 1)
trend_fn           = np.poly1d(coeffs)
trend_mm_per_month = coeffs[0] * 30
trend_line         = trend_fn(s149_ts['days'])

s149_coh = ts[ts['segment_id']==149].dropna(subset=['coherence'])\
             .sort_values('date').copy()
s149_coh['date_dt'] = pd.to_datetime(s149_coh['date'])

s149_row   = df[df['segment_id']==149].iloc[0]
all_trends = df[['segment_id','mean_slope_deg','risk_score',
                  'risk_category','disp_trend_mm_per_month']].copy()

print(f"  Trend: {trend_mm_per_month:.3f} mm/month")
print(f"  Pairs: {len(s149_ts)}")

RISK_COLORS = {'Low':GREEN,'Moderate':BLUE,'High':ORANGE,'Critical':RED}

fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
fig.suptitle(
    'Segment 149 — I-40 Mountain Corridor\n'
    'InSAR-Detected Subsidence: A Case Study in Satellite-Driven Risk Detection',
    color=TEXT, fontsize=16, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97,
                       top=0.93, bottom=0.06)

# ── Panel 1: Displacement time series ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(DARK_BG)
seasons = {
    'Winter': (BLUE,   [12,1,2]),
    'Spring': (GREEN,  [3,4,5]),
    'Summer': (ORANGE, [6,7,8]),
    'Autumn': (RED,    [9,10,11]),
}
for season, (color, months) in seasons.items():
    mask = s149_ts['date_dt'].dt.month.isin(months)
    ax1.scatter(s149_ts.loc[mask,'date_dt'], s149_ts.loc[mask,'disp_mm'],
                color=color, s=60, zorder=4, label=season, alpha=0.9)
ax1.plot(s149_ts['date_dt'], trend_line,
         color=RED, linewidth=2.5, linestyle='--', zorder=3,
         label=f'Linear trend: {trend_mm_per_month:.3f} mm/month')
ax1.plot(s149_ts['date_dt'], s149_ts['disp_mm'],
         color=MUTED, linewidth=0.8, alpha=0.4, zorder=2)
ax1.axhline(0, color=MUTED, linestyle='-', linewidth=1, alpha=0.3)
peak_idx = s149_ts['disp_mm'].idxmin()
ax1.annotate(
    f'Peak subsidence\n{s149_ts["disp_mm"].min():.1f} mm',
    xy=(s149_ts.loc[peak_idx,'date_dt'], s149_ts['disp_mm'].min()),
    xytext=(pd.Timestamp('2022-08-01'), s149_ts['disp_mm'].min()-3),
    color=RED, fontsize=9,
    arrowprops=dict(arrowstyle='->', color=RED, lw=1.5)
)
ax1.set_title(
    '2-Year InSAR Displacement Time Series — Segment 149\n'
    'Sentinel-1, ASF HyP3, 35 pairs, January 2022 – January 2024',
    color=TEXT, fontsize=12, fontweight='bold')
ax1.set_xlabel('Date', color=MUTED, fontsize=10)
ax1.set_ylabel('Vertical Displacement (mm)', color=MUTED, fontsize=10)
ax1.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT,
           fontsize=9, ncol=5, loc='upper right')
ax1.tick_params(colors=MUTED)
ax1.grid(color='#2d3147', alpha=0.4)
for s in ax1.spines.values(): s.set_edgecolor('#2d3147')

# ── Panel 2: Coherence time series ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(DARK_BG)
ax2.plot(s149_coh['date_dt'], s149_coh['coherence'],
         color=BLUE, linewidth=1.5, marker='o', markersize=4)
ax2.axhline(0.3, color=RED, linestyle='--', linewidth=1.5, alpha=0.7,
            label='Low coherence threshold')
mean_coh = s149_coh['coherence'].mean()
ax2.axhline(mean_coh, color=ACCENT, linestyle='--', linewidth=1.2, alpha=0.7,
            label=f'Mean: {mean_coh:.3f}')
ax2.fill_between(s149_coh['date_dt'], s149_coh['coherence'], 0.3,
                 where=s149_coh['coherence']>0.3, color=BLUE, alpha=0.15)
ax2.set_title('InSAR Coherence Over Time', color=TEXT, fontsize=10, fontweight='bold')
ax2.set_xlabel('Date', color=MUTED, fontsize=8)
ax2.set_ylabel('Coherence', color=MUTED, fontsize=8)
ax2.set_ylim(0, 1)
ax2.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=7)
ax2.tick_params(colors=MUTED, labelsize=7)
ax2.grid(color='#2d3147', alpha=0.4)
for s in ax2.spines.values(): s.set_edgecolor('#2d3147')

# ── Panel 3: Slope vs Trend scatter ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(DARK_BG)
for cat, color in RISK_COLORS.items():
    sub = all_trends[all_trends['risk_category']==cat]
    if len(sub)==0: continue
    ax3.scatter(sub['mean_slope_deg'], sub['disp_trend_mm_per_month'],
                color=color, s=15, alpha=0.6, label=cat)
ax3.scatter(s149_row['mean_slope_deg'], s149_row['disp_trend_mm_per_month'],
            color='white', s=200, marker='*', zorder=5, label='Segment 149')
ax3.annotate('Segment 149\nLow slope\nHigh trend',
             xy=(s149_row['mean_slope_deg'], s149_row['disp_trend_mm_per_month']),
             xytext=(8, -1.2), color='white', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='white', lw=1.2))
ax3.axhline(0, color=MUTED, linestyle='--', linewidth=1, alpha=0.4)
ax3.set_title('Slope vs Displacement Trend\n(All I-40 segments)',
              color=TEXT, fontsize=10, fontweight='bold')
ax3.set_xlabel('Mean Slope (degrees)', color=MUTED, fontsize=8)
ax3.set_ylabel('Trend (mm/month)', color=MUTED, fontsize=8)
ax3.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=7)
ax3.tick_params(colors=MUTED, labelsize=7)
ax3.grid(color='#2d3147', alpha=0.4)
for s in ax3.spines.values(): s.set_edgecolor('#2d3147')

# ── Panel 4: Key metrics ──────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor(CARD_BG)
ax4.axis('off')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.text(0.5, 0.97, 'Segment 149 — Key Metrics',
         ha='center', va='top', color=ACCENT,
         fontsize=11, fontweight='bold', transform=ax4.transAxes)

metrics = [
    ('Segment ID',        '149',                         ACCENT),
    ('Corridor',          'I-40',                        TEXT),
    ('Rock Class',        'Carbonate',                   ORANGE),
    ('Mean Slope',        '5.2 degrees',                 GREEN),
    ('Elevation Range',   '254.6 m',                     TEXT),
    ('InSAR Pairs',       '32 of 35',                    TEXT),
    ('Mean Displacement', '-3.318 mm',                   ORANGE),
    ('Peak Displacement', f'{s149_ts["disp_mm"].min():.1f} mm', RED),
    ('Trend',             '-1.785 mm/month',             RED),
    ('2yr Cumulative',    f'~{trend_mm_per_month*24:.0f} mm', RED),
    ('Mean Coherence',    f'{mean_coh:.3f}',             BLUE),
    ('Terrain Only',      'LOW RISK',                    GREEN),
    ('With InSAR',        'HIGH RISK',                   RED),
]
for i, (label, value, color) in enumerate(metrics):
    y = 0.88 - i * 0.062
    ax4.text(0.03, y, label + ':',
             color=MUTED, fontsize=8, va='top', transform=ax4.transAxes)
    ax4.text(0.97, y, value,
             color=color, fontsize=8.5, va='top',
             ha='right', fontweight='bold', transform=ax4.transAxes)
for s in ax4.spines.values(): s.set_edgecolor('#2d3147')

# ── Panel 5: Corridor map ─────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0:2])
ax5.set_facecolor(DARK_BG)
for cat, color in RISK_COLORS.items():
    sub_segs = segs[segs['segment_id'].isin(
        df[df['risk_category']==cat]['segment_id'])]
    if len(sub_segs)==0: continue
    sub_segs.plot(ax=ax5, color=color, linewidth=1.5, alpha=0.7, label=cat)
seg149_geom = segs[segs['segment_id']==149]
seg149_geom.plot(ax=ax5, color='white', linewidth=6, alpha=0.9, zorder=5)
seg149_geom.plot(ax=ax5, color=RED, linewidth=3.5, zorder=6,
                 label='Segment 149')
centroid = seg149_geom.geometry.iloc[0].centroid
ax5.annotate('Segment 149\n-1.785 mm/mo',
             xy=(centroid.x, centroid.y),
             xytext=(centroid.x - 0.8, centroid.y + 0.12),
             color='white', fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
             bbox=dict(boxstyle='round', facecolor=CARD_BG,
                       edgecolor=RED, alpha=0.9))
ax5.set_title('Segment 149 Location — I-40 Corridor',
              color=TEXT, fontsize=10, fontweight='bold')
ax5.set_xlabel('Longitude', color=MUTED, fontsize=8)
ax5.set_ylabel('Latitude',  color=MUTED, fontsize=8)
ax5.tick_params(colors=MUTED, labelsize=7)
ax5.legend(facecolor=CARD_BG, edgecolor=MUTED,
           labelcolor=TEXT, fontsize=8, loc='lower right')
for s in ax5.spines.values(): s.set_edgecolor('#2d3147')

# ── Panel 6: Key message ──────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor(CARD_BG)
ax6.axis('off')
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

messages = [
    ('WHY THIS MATTERS',                  ACCENT, 12, 'bold'),
    ('',                                  TEXT,    4, 'normal'),
    ('Segment 149 has only 5.2 deg slope.', TEXT,  9, 'normal'),
    ('A terrain-only model would',         TEXT,   9, 'normal'),
    ('classify it as LOW RISK',            GREEN, 10, 'bold'),
    ('and skip inspection.',               TEXT,   9, 'normal'),
    ('',                                  TEXT,    4, 'normal'),
    ('Sentinel-1 InSAR detects',           TEXT,   9, 'normal'),
    ('-1.785 mm/month sustained',          RED,   11, 'bold'),
    ('subsidence over 2 years.',           TEXT,   9, 'normal'),
    ('Model correctly elevates it',        TEXT,   9, 'normal'),
    ('to HIGH risk.',                      RED,   10, 'bold'),
    ('',                                  TEXT,    4, 'normal'),
    ('Core value of satellite InSAR:',     ACCENT, 9, 'bold'),
    ('Catch what eyes cannot see.',        ACCENT,10, 'bold'),
]
y_pos = 0.96
for text, color, size, weight in messages:
    ax6.text(0.5, y_pos, text, ha='center', va='top',
             color=color, fontsize=size, fontweight=weight,
             transform=ax6.transAxes)
    y_pos -= size * 0.007 + 0.02

for spine in ax6.spines.values():
    spine.set_edgecolor(RED)
    spine.set_linewidth(2)

plt.savefig('outputs/figures/fig12_segment149_deepdive.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Saved: outputs/figures/fig12_segment149_deepdive.png")
print("Segment 149 deep dive complete.")
