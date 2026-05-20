import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

DARK_BG = '#0f1117'
CARD_BG = '#1e2235'
ACCENT  = '#64ffda'
TEXT    = '#e0e0e0'
MUTED   = '#8892b0'
plt.style.use('dark_background')

# ── 1. Load all real features ─────────────────────────────────────────────────
print("Loading real feature data...")
terrain  = pd.read_csv('data/processed/terrain_features/terrain_features.csv')
insar    = pd.read_csv('data/processed/insar_features/insar_features_extended.csv')
geology  = pd.read_csv('data/raw/geology/geology_per_segment.csv')
traffic  = pd.read_csv('data/raw/traffic/aadt_per_segment.csv')

df = terrain.merge(insar,   on='segment_id', how='left')
df = df.merge(geology, on='segment_id', how='left')
df = df.merge(traffic, on='segment_id', how='left')
df['rock_class'] = df['rock_class'].fillna('other')
df['corridor']   = 'I-40'
print(f"  Segments: {len(df)}")

# Rock risk encoding
rock_risk = {'metamorphic_crystalline':4,'clastic_sedimentary':3,
             'unconsolidated':3,'mixed_sedimentary':2,'carbonate':2,'other':1}
df['rock_risk_score'] = df['rock_class'].map(rock_risk).fillna(1)

# ── 2. Select clustering features (only real measurements) ────────────────────
# CRITICAL: no proxy labels, no derived risk scores — only raw measurements
cluster_features = [
    # Terrain
    'mean_slope_deg',
    'max_slope_deg',
    'elev_range_m',
    'terrain_roughness',
    'mean_curvature',
    # InSAR (real satellite)
    'mean_disp_mm',
    'disp_trend_mm_per_month',
    'disp_std_mm',
    'mean_coherence',
    # Geology
    'rock_risk_score',
    # Traffic
    'AADT_2022',
    'truck_pct',
]

X_raw = df[cluster_features].fillna(0)

# ── 3. Standardize features ───────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ── 4. Find optimal K using silhouette score ──────────────────────────────────
print("\nFinding optimal number of clusters...")
sil_scores = {}
for k in range(2, 9):
    km  = KMeans(n_clusters=k, random_state=42, n_init=20)
    lbl = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, lbl)
    sil_scores[k] = sil
    print(f"  K={k}  silhouette={sil:.3f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n  Best K: {best_k} (silhouette={sil_scores[best_k]:.3f})")

# ── 5. Final clustering with best K ──────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df['cluster'] = km_final.fit_predict(X_scaled)

# ── 6. Characterize each cluster ──────────────────────────────────────────────
print("\nCluster characteristics:")
cluster_summary = df.groupby('cluster')[cluster_features].mean().round(3)
print(cluster_summary.to_string())

# ── 7. Assign risk tier to each cluster based on real measurements ────────────
# Rank clusters by a composite of:
# slope + displacement magnitude + trend + roughness (all real measurements)
def norm(s):
    mn,mx = s.min(),s.max()
    return (s-mn)/(mx-mn) if mx>mn else s*0

cluster_risk_score = (
    0.25 * norm(cluster_summary['mean_slope_deg'])             +
    0.20 * norm(cluster_summary['elev_range_m'])               +
    0.20 * norm(abs(cluster_summary['mean_disp_mm']))          +
    0.20 * norm(abs(cluster_summary['disp_trend_mm_per_month']))+
    0.15 * norm(cluster_summary['terrain_roughness'])
)

# Rank clusters: 0=lowest risk, best_k-1=highest risk
cluster_rank = cluster_risk_score.rank().astype(int) - 1
risk_labels  = {0:'Low', 1:'Moderate', 2:'High', 3:'Critical'}

# Map cluster rank to risk category
n_cats = min(best_k, 4)
boundaries = np.linspace(0, best_k, n_cats+1)
def rank_to_cat(rank):
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if lo <= rank < hi:
            return list(risk_labels.values())[i]
    return 'High'

cluster_to_risk = {}
for cluster_id, rank in cluster_rank.items():
    cluster_to_risk[cluster_id] = rank_to_cat(rank)

df['risk_category_unsup'] = df['cluster'].map(cluster_to_risk)

print(f"\nCluster to risk mapping:")
for cid, risk in cluster_to_risk.items():
    n = (df['cluster']==cid).sum()
    slope = cluster_summary.loc[cid,'mean_slope_deg']
    disp  = cluster_summary.loc[cid,'mean_disp_mm']
    trend = cluster_summary.loc[cid,'disp_trend_mm_per_month']
    print(f"  Cluster {cid} → {risk:10s} "
          f"(n={n:3d}, slope={slope:.1f}°, "
          f"disp={disp:.2f}mm, trend={trend:.3f}mm/mo)")

print(f"\nRisk distribution (unsupervised):")
print(df['risk_category_unsup'].value_counts().to_string())

# ── 8. PCA for visualization ──────────────────────────────────────────────────
print("\nRunning PCA for cluster visualization...")
pca     = PCA(n_components=2, random_state=42)
X_pca   = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]
print(f"  PCA variance explained: "
      f"{pca.explained_variance_ratio_[0]:.1%} + "
      f"{pca.explained_variance_ratio_[1]:.1%} = "
      f"{sum(pca.explained_variance_ratio_):.1%}")

# ── 9. Train RF on unsupervised labels ────────────────────────────────────────
print("\nTraining Random Forest on unsupervised cluster labels...")
from sklearn.preprocessing import LabelEncoder
df['risk_enc'] = LabelEncoder().fit_transform(df['risk_category_unsup'])

rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                             class_weight='balanced', random_state=42)
cv = cross_val_score(rf, X_raw.fillna(0), df['risk_enc'],
                     cv=5, scoring='f1_weighted')
print(f"  CV F1 (weighted): {cv.mean():.3f} +/- {cv.std():.3f}")
rf.fit(X_raw.fillna(0), df['risk_enc'])
df['rf_prob_unsup'] = rf.predict_proba(X_raw.fillna(0)).max(axis=1)

# ── 10. Composite unsupervised risk score ─────────────────────────────────────
def normdf(s):
    s = pd.to_numeric(s,errors='coerce').fillna(0)
    mn,mx = s.min(),s.max()
    return (s-mn)/(mx-mn) if mx>mn else pd.Series(np.zeros(len(s)))

terrain_susc = (0.4*normdf(df['mean_slope_deg']) +
                0.3*normdf(df['elev_range_m'])    +
                0.3*normdf(df['terrain_roughness']))
insar_signal = (0.4*normdf(abs(df['mean_disp_mm'])) +
                0.4*normdf(abs(df['disp_trend_mm_per_month'])) +
                0.2*normdf(df['disp_std_mm']))
road_exp     = (0.6*normdf(df['AADT_2022']) +
                0.4*normdf(df['truck_AADT'].fillna(0)))
geo_fac      = normdf(df['rock_risk_score'])

df['risk_score_unsup'] = (
    0.35 * df['rf_prob_unsup'] +
    0.25 * terrain_susc        +
    0.20 * insar_signal        +
    0.12 * road_exp            +
    0.08 * geo_fac
).round(4)

# ── 11. Generate figures ───────────────────────────────────────────────────────
RISK_COLORS = {'Low':'#26de81','Moderate':'#4a90d9',
               'High':'#ff9f43','Critical':'#ff4b4b'}
CLUSTER_COLORS = ['#64ffda','#ff9f43','#ff4b4b','#4a90d9',
                  '#26de81','#a29bfe','#fd79a8']

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=DARK_BG)
fig.suptitle('Unsupervised Slope Risk Clustering — I-40 Corridor\n'
             'K-Means on Real Features Only (No Proxy Labels)',
             color=TEXT, fontsize=14, fontweight='bold')

# Panel A: PCA scatter by cluster
ax = axes[0,0]
ax.set_facecolor(DARK_BG)
for cid in range(best_k):
    sub = df[df['cluster']==cid]
    ax.scatter(sub['pca1'], sub['pca2'],
               color=CLUSTER_COLORS[cid], s=25, alpha=0.8,
               label=f'Cluster {cid} ({cluster_to_risk[cid]})')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
              color=MUTED)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
              color=MUTED)
ax.set_title('PCA — Cluster Separation', color=TEXT, fontweight='bold')
ax.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)
ax.tick_params(colors=MUTED)
ax.grid(color='#2d3147', alpha=0.4)
for s in ax.spines.values(): s.set_edgecolor('#2d3147')

# Panel B: PCA scatter by risk category
ax = axes[0,1]
ax.set_facecolor(DARK_BG)
for cat, color in RISK_COLORS.items():
    sub = df[df['risk_category_unsup']==cat]
    if len(sub)==0: continue
    ax.scatter(sub['pca1'], sub['pca2'],
               color=color, s=25, alpha=0.8, label=cat)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
              color=MUTED)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
              color=MUTED)
ax.set_title('PCA — Risk Category Assignment', color=TEXT, fontweight='bold')
ax.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
ax.tick_params(colors=MUTED)
ax.grid(color='#2d3147', alpha=0.4)
for s in ax.spines.values(): s.set_edgecolor('#2d3147')

# Panel C: Cluster feature heatmap
ax = axes[1,0]
ax.set_facecolor(DARK_BG)
features_plot = ['mean_slope_deg','elev_range_m','terrain_roughness',
                 'mean_disp_mm','disp_trend_mm_per_month','mean_coherence',
                 'rock_risk_score','AADT_2022']
labels_plot   = ['Slope°','Elev Range','Roughness',
                 'Displacement','Trend mm/mo','Coherence',
                 'Rock Risk','AADT']
heatmap_data  = cluster_summary[features_plot].copy()
# Normalize each feature 0-1 for display
for col in heatmap_data.columns:
    mn,mx = heatmap_data[col].min(), heatmap_data[col].max()
    if mx>mn:
        heatmap_data[col] = (heatmap_data[col]-mn)/(mx-mn)

im = ax.imshow(heatmap_data.values.T, cmap='RdYlGn_r',
               aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(best_k))
ax.set_xticklabels([f'C{i}\n({cluster_to_risk[i]})' for i in range(best_k)],
                   color=TEXT, fontsize=9)
ax.set_yticks(range(len(labels_plot)))
ax.set_yticklabels(labels_plot, color=TEXT, fontsize=9)
ax.set_title('Cluster Feature Profile (normalized)',
             color=TEXT, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8).ax.yaxis.set_tick_params(color=MUTED)
for s in ax.spines.values(): s.set_edgecolor('#2d3147')

# Panel D: Risk score distribution comparison
ax = axes[1,1]
ax.set_facecolor(DARK_BG)
df_old = pd.read_csv('data/processed/model_table/full_feature_table.csv')
ax.hist(df_old['risk_score'].dropna(), bins=30,
        color=ACCENT, alpha=0.6, label='Proxy labels (old)',
        edgecolor='none', density=True)
ax.hist(df['risk_score_unsup'], bins=30,
        color='#ff9f43', alpha=0.6, label='Unsupervised (new)',
        edgecolor='none', density=True)
ax.set_xlabel('Risk Score', color=MUTED)
ax.set_ylabel('Density',    color=MUTED)
ax.set_title('Risk Score: Proxy vs Unsupervised',
             color=TEXT, fontweight='bold')
ax.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
ax.tick_params(colors=MUTED)
ax.grid(color='#2d3147', alpha=0.4)
for s in ax.spines.values(): s.set_edgecolor('#2d3147')

plt.tight_layout()
plt.savefig('outputs/figures/fig11_unsupervised_clustering.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("\nSaved: outputs/figures/fig11_unsupervised_clustering.png")

# ── 12. Save results ──────────────────────────────────────────────────────────
df[['segment_id','cluster','risk_category_unsup',
    'risk_score_unsup','pca1','pca2']]\
  .to_csv('data/processed/model_table/unsupervised_clusters.csv', index=False)

print(f"\nFinal unsupervised risk distribution:")
print(df['risk_category_unsup'].value_counts().sort_index().to_string())
print(f"\nTop 5 highest-risk segments (unsupervised):")
print(df.nlargest(5,'risk_score_unsup')[
    ['segment_id','risk_score_unsup','risk_category_unsup',
     'mean_slope_deg','mean_disp_mm','disp_trend_mm_per_month',
     'rock_class']].round(3).to_string(index=False))
print(f"\nSegment 149 (unsupervised):")
s149 = df[df.segment_id==149].iloc[0]
print(f"  Cluster:       {int(s149.cluster)}")
print(f"  Risk category: {s149.risk_category_unsup}")
print(f"  Risk score:    {s149.risk_score_unsup:.3f}")
print(f"  Trend:         {s149.disp_trend_mm_per_month:.3f} mm/month")
print("\nPhase 9 (unsupervised clustering) complete.")
