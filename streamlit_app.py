import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="TDOT Slope Risk Monitor",
    page_icon="🏔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #ffffff !important; color: #1a1a2e !important; }
    section[data-testid="stSidebar"] { background-color: #f0f4f8 !important; }
    section[data-testid="stSidebar"] * { color: #1a1a2e !important; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%) !important;
        border: 1px solid #dde3ea !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stMetricValue"] { color: #1a1a2e !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #5a6a7a !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #eef2f7 !important; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #5a6a7a !important; border-radius: 6px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #1a73e8 !important; color: #ffffff !important; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1a73e8;
        border-bottom: 2px solid #1a73e8; padding-bottom: 8px; margin-bottom: 16px;
    }
    hr { border-color: #dde3ea !important; }
    .stDataFrame { border-radius: 8px; border: 1px solid #dde3ea; }
    div[data-testid="stVerticalBlock"] { background-color: #ffffff; }
    .main { background-color: #ffffff !important; }
    h1, h2, h3, p, span, div { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {
    'Low':'#26de81','Moderate':'#4a90d9',
    'High':'#ff9f43','Critical':'#ff4b4b'
}
CORRIDOR_COLORS = {'I-40':'#64ffda', 'I-75':'#ff9f43'}

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df   = pd.read_csv('data/processed/model_table/combined_corridors.csv')
    i40  = gpd.read_file('data/processed/corridor/segments.geojson')
    i75  = gpd.read_file('data/processed/corridor/i75_segments.geojson')
    i40['segment_id'] = i40['segment_id'].astype(int)
    i75['segment_id'] = i75['segment_id'].astype(int) + 1000
    i40['corridor']   = 'I-40'
    i75['corridor']   = 'I-75'
    gdf = pd.concat([i40, i75], ignore_index=True)
    return df, gdf

df, gdf = load_data()

# Merge attributes into geodataframe for map
map_df = gdf[['segment_id','corridor','geometry']].copy()
map_df = map_df.merge(
    df[['segment_id','risk_score','risk_category',
        'mean_slope_deg','mean_disp_mm','rock_class',
        'AADT_2022','mean_coherence']],
    on='segment_id', how='left'
)
map_df['risk_category'] = map_df['risk_category'].fillna('Low')
map_df['rock_class']    = map_df['rock_class'].fillna('other')

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏔 TDOT Slope Monitor")
    st.markdown("*East Tennessee · Multi-Corridor*")
    st.divider()
    st.markdown("### Filters")
    corridor_filter = st.multiselect(
        "Corridor", ['I-40','I-75'], default=['I-40','I-75'])
    risk_filter = st.multiselect(
        "Risk Category",
        ['Low','Moderate','High','Critical'],
        default=['Low','Moderate','High','Critical'])
    slope_range = st.slider(
        "Mean Slope (°)", 0.0,
        float(df['mean_slope_deg'].max()),
        (0.0, float(df['mean_slope_deg'].max())))
    rock_options = sorted(df['rock_class'].dropna().unique().tolist())
    rock_filter  = st.multiselect("Rock Class", rock_options, default=rock_options)
    aadt_min     = st.slider("Min AADT", 0, int(df['AADT_2022'].max()), 0, step=5000)
    st.divider()
    st.markdown("### Corridors")
    st.markdown("""
- 🟦 **I-40** — 342 segments · 274 km
  Cookeville → NC Border
- 🟧 **I-75** — 90 segments · 72 km
  Jellico Mountain Section
    """)
    st.divider()
    st.markdown("### Data Sources")
    st.markdown("""
- 🛰 Sentinel-1 InSAR · ASF HyP3
- 🏔 USGS 3DEP 30m DEM
- 🪨 USGS State Geology TN
- 🚗 FHWA HPMS 2022
    """)

filtered = df[
    df['corridor'].isin(corridor_filter) &
    df['risk_category'].isin(risk_filter) &
    df['mean_slope_deg'].between(*slope_range) &
    df['rock_class'].isin(rock_filter) &
    (df['AADT_2022'] >= aadt_min)
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#1a73e8;margin-bottom:0'>
    TDOT Slope Movement Monitoring
</h1>
<p style='color:#5a6a7a;margin-top:4px;font-size:1.05rem'>
    Multi-Corridor System · I-40 + I-75 · East Tennessee ·
    Sentinel-1 InSAR + USGS Terrain + Geology + FHWA Traffic
</p>
""", unsafe_allow_html=True)
st.divider()

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
high_crit = int((df['risk_category'].isin(['High','Critical'])).sum())
k1.metric("Total Segments",     f"{len(df):,}")
k2.metric("Corridors",          "2",
          delta="I-40 + I-75")
k3.metric("High/Critical Risk", f"{high_crit}",
          delta=f"{high_crit/len(df)*100:.0f}% of network")
k4.metric("Avg Risk Score",     f"{df['risk_score'].mean():.3f}")
k5.metric("Max Displacement",   f"{df['mean_disp_mm'].min():.1f} mm",
          delta="Real satellite · Seg 149")
k6.metric("Mean Coherence",     f"{df['mean_coherence'].mean():.3f}")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🗺 Risk Map",
    "📊 Analysis",
    "🪨 Geology & InSAR",
    "🛣 Corridor Compare",
    "📋 Segment Table",
    "🔬 Risk Clustering"
])

# ══ TAB 1: MAP ════════════════════════════════════════════════════════════════
with tab1:
    col_map, col_right = st.columns([3,2])
    with col_map:
        st.markdown('<div class="section-header">Interactive Risk Map — I-40 + I-75</div>',
                    unsafe_allow_html=True)
        m = folium.Map(location=[36.1,-84.2], zoom_start=7,
                       tiles='CartoDB dark_matter')

        for _, row in map_df.iterrows():
            if row.geometry is None: continue
            cat    = str(row['risk_category'])
            color  = RISK_COLORS.get(cat,'#8892b0')
            score  = row['risk_score'] if pd.notna(row['risk_score']) else 0
            sid    = int(row['segment_id'])
            slope  = row['mean_slope_deg'] if pd.notna(row['mean_slope_deg']) else 0
            disp   = row['mean_disp_mm']   if pd.notna(row['mean_disp_mm'])   else 0
            rock   = str(row['rock_class'])
            aadt   = int(row['AADT_2022']) if pd.notna(row['AADT_2022']) else 0
            corr   = str(row['corridor'])
            weight = {'Low':3,'Moderate':5,'High':6,'Critical':8}.get(cat,3)
            try:
                coords = [[c[1],c[0]] for c in row.geometry.coords]
            except:
                continue
            folium.PolyLine(
                coords, color=color, weight=weight, opacity=0.9,
                tooltip=folium.Tooltip(
                    f"<b style='color:{color}'>{corr} · Seg {sid} — {cat}</b><br>"
                    f"Risk: <b>{score:.3f}</b><br>"
                    f"Slope: {slope:.1f}° | Disp: {disp:.2f} mm<br>"
                    f"Rock: {rock} | AADT: {aadt:,}"
                )
            ).add_to(m)

        # Corridor labels
        folium.Marker([35.95,-84.5],
            icon=folium.DivIcon(html="<div style='color:#1a73e8;font-weight:bold;font-size:13px'>I-40</div>")
        ).add_to(m)
        folium.Marker([36.35,-84.0],
            icon=folium.DivIcon(html="<div style='color:#ff9f43;font-weight:bold;font-size:13px'>I-75</div>")
        ).add_to(m)

        legend = """
        <div style='position:fixed;bottom:20px;left:20px;z-index:1000;
                    background:#f0f4f8;padding:12px 16px;border-radius:10px;
                    border:1px solid #2d3147;color:#1a1a2e;font-size:13px'>
        <b style='color:#1a73e8'>Risk Level</b><br><br>
        <span style='color:#26de81;font-size:18px'>━━</span> Low<br>
        <span style='color:#4a90d9;font-size:18px'>━━</span> Moderate<br>
        <span style='color:#ff9f43;font-size:18px'>━━</span> High<br>
        <span style='color:#ff4b4b;font-size:18px'>━━</span> Critical
        </div>"""
        m.get_root().html.add_child(folium.Element(legend))
        st_folium(m, width=None, height=520, returned_objects=[])

    with col_right:
        st.markdown('<div class="section-header">Top 10 Highest-Risk Segments</div>',
                    unsafe_allow_html=True)
        top10 = df.nlargest(10,'risk_score')[
            ['segment_id','corridor','risk_score','risk_category',
             'mean_slope_deg','mean_disp_mm','rock_class','AADT_2022']
        ].round(3)
        top10.columns = ['Seg','Road','Score','Cat','Slope°','Disp mm','Rock','AADT']
        st.dataframe(top10, use_container_width=True, hide_index=True,
            column_config={
                'Score': st.column_config.ProgressColumn(
                    'Score',min_value=0,max_value=1,format='%.3f'),
            })

        st.markdown('<div class="section-header">Risk Distribution</div>',
                    unsafe_allow_html=True)
        cat_df = df['risk_category'].value_counts()\
                   .reindex(['Low','Moderate','High','Critical'],fill_value=0)\
                   .reset_index()
        cat_df.columns = ['Category','Count']
        fig_d = px.pie(cat_df,names='Category',values='Count',
                       color='Category',color_discrete_map=RISK_COLORS,hole=0.55)
        fig_d.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(font=dict(color='#e0e0e0')),height=250)
        fig_d.update_traces(textfont_color='white')
        st.plotly_chart(fig_d,use_container_width=True)

# ══ TAB 2: ANALYSIS ═══════════════════════════════════════════════════════════
with tab2:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Risk Score Along Corridors</div>',
                    unsafe_allow_html=True)
        fig_l = go.Figure()
        for corridor in ['I-40','I-75']:
            sub = df[df['corridor']==corridor].sort_values('segment_id')
            fig_l.add_trace(go.Scatter(
                x=list(range(len(sub))), y=sub['risk_score'],
                mode='lines', name=corridor,
                line=dict(color=CORRIDOR_COLORS[corridor],width=2)))
        fig_l.add_hline(y=0.50,line_dash='dash',line_color='#8892b0',
                        annotation_text='High threshold',
                        annotation_font_color='#8892b0')
        fig_l.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',height=320,
            xaxis=dict(title='Segment Position',gridcolor='#dde3ea'),
            yaxis=dict(title='Risk Score',gridcolor='#dde3ea'),
            legend=dict(bgcolor='#ffffff'),margin=dict(t=10,b=10))
        st.plotly_chart(fig_l,use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Feature Importance</div>',
                    unsafe_allow_html=True)
        fi = pd.read_csv('outputs/tables/feature_importance.csv',
                         names=['feature','importance'],header=0)
        fi = fi.sort_values('importance',ascending=True).tail(10)
        fig_fi = go.Figure(go.Bar(
            x=fi['importance'],y=fi['feature'],orientation='h',
            marker=dict(color=fi['importance'],
                colorscale=[[0,'#2d3147'],[0.5,'#4a90d9'],[1,'#64ffda']]),
            text=fi['importance'].round(3),
            textposition='outside',textfont=dict(color='#e0e0e0')))
        fig_fi.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',height=320,
            xaxis=dict(title='Importance',gridcolor='#dde3ea'),
            yaxis=dict(gridcolor='#dde3ea'),margin=dict(t=10,b=10))
        st.plotly_chart(fig_fi,use_container_width=True)

    st.divider()
    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header">Slope vs Displacement</div>',
                    unsafe_allow_html=True)
        fig_sc = px.scatter(df,x='mean_slope_deg',y='mean_disp_mm',
            color='risk_category',color_discrete_map=RISK_COLORS,
            symbol='corridor',size='elev_range_m',size_max=12,
            hover_data=['segment_id','corridor','rock_class'],
            labels={'mean_slope_deg':'Mean Slope (°)',
                    'mean_disp_mm':'Mean Displacement (mm)'})
        fig_sc.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',height=320,
            xaxis=dict(gridcolor='#dde3ea'),yaxis=dict(gridcolor='#dde3ea'),
            legend=dict(bgcolor='#ffffff'),margin=dict(t=10,b=10))
        st.plotly_chart(fig_sc,use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Risk Score vs AADT</div>',
                    unsafe_allow_html=True)
        fig_a = px.scatter(df,x='AADT_2022',y='risk_score',
            color='risk_category',color_discrete_map=RISK_COLORS,
            symbol='corridor',size='truck_pct',size_max=12,
            hover_data=['segment_id','corridor','rock_class'],
            labels={'AADT_2022':'AADT 2022','risk_score':'Risk Score'})
        fig_a.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',height=320,
            xaxis=dict(gridcolor='#dde3ea'),yaxis=dict(gridcolor='#dde3ea'),
            legend=dict(bgcolor='#ffffff'),margin=dict(t=10,b=10))
        st.plotly_chart(fig_a,use_container_width=True)

# ══ TAB 3: GEOLOGY & INSAR ════════════════════════════════════════════════════
with tab3:
    c5,c6 = st.columns(2)
    with c5:
        st.markdown('<div class="section-header">Mean Risk by Rock Type</div>',
                    unsafe_allow_html=True)
        rock_agg = df.groupby('rock_class').agg(
            mean_risk=('risk_score','mean'),
            count=('segment_id','count')).reset_index()\
            .sort_values('mean_risk',ascending=True)
        fig_r = go.Figure(go.Bar(
            x=rock_agg['mean_risk'],y=rock_agg['rock_class'],orientation='h',
            marker=dict(color=rock_agg['mean_risk'],
                colorscale=[[0,'#26de81'],[0.5,'#ff9f43'],[1,'#ff4b4b']]),
            text=[f"n={c}" for c in rock_agg['count']],
            textposition='outside',textfont=dict(color='#e0e0e0')))
        fig_r.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',height=300,
            xaxis=dict(title='Mean Risk Score',gridcolor='#dde3ea'),
            yaxis=dict(gridcolor='#dde3ea'),margin=dict(t=10,b=10))
        st.plotly_chart(fig_r,use_container_width=True)

    with c6:
        st.markdown('<div class="section-header">InSAR Coherence Distribution</div>',
                    unsafe_allow_html=True)
        fig_coh = px.histogram(df,x='mean_coherence',nbins=30,
            color='corridor',color_discrete_map=CORRIDOR_COLORS,
            barmode='overlay',opacity=0.75,
            labels={'mean_coherence':'Mean Coherence'})
        fig_coh.add_vline(x=0.3,line_dash='dash',line_color='#ff4b4b',
            annotation_text='Low coherence threshold',
            annotation_font_color='#ff4b4b')
        fig_coh.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',height=300,
            xaxis=dict(gridcolor='#dde3ea'),
            yaxis=dict(title='Count',gridcolor='#dde3ea'),
            legend=dict(bgcolor='#ffffff'),margin=dict(t=10,b=10))
        st.plotly_chart(fig_coh,use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">Displacement by Rock Class (both corridors)</div>',
                unsafe_allow_html=True)
    fig_box = px.box(df,x='rock_class',y='mean_disp_mm',
        color='corridor',color_discrete_map=CORRIDOR_COLORS,
        points='outliers',
        labels={'rock_class':'Rock Class','mean_disp_mm':'Mean Displacement (mm)'})
    fig_box.update_layout(
        paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
        font_color='#1a1a2e',height=320,
        xaxis=dict(gridcolor='#dde3ea'),yaxis=dict(gridcolor='#dde3ea'),
        legend=dict(bgcolor='#ffffff'),margin=dict(t=10,b=10))
    st.plotly_chart(fig_box,use_container_width=True)

# ══ TAB 4: CORRIDOR COMPARE ═══════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">I-40 vs I-75 — Side by Side Comparison</div>',
                unsafe_allow_html=True)

    metrics = ['mean_slope_deg','elev_range_m','terrain_roughness',
               'mean_disp_mm','mean_coherence','risk_score','AADT_2022','truck_pct']
    labels  = ['Mean Slope (°)','Elev Range (m)','Roughness',
               'Displacement (mm)','Coherence','Risk Score','AADT','Truck %']

    comp = df.groupby('corridor')[metrics].mean().round(3).T
    comp.index = labels

    c_a,c_b = st.columns(2)
    with c_a:
        st.markdown("**I-40 vs I-75 — Mean Values**")
        st.dataframe(comp.style.background_gradient(cmap='RdYlGn_r',axis=1),
                     use_container_width=True)

    with c_b:
        st.markdown("**Risk Category Breakdown**")
        rc = df.groupby(['corridor','risk_category']).size().reset_index(name='count')
        fig_rc = px.bar(rc,x='corridor',y='count',color='risk_category',
                        color_discrete_map=RISK_COLORS,barmode='group',
                        labels={'count':'Segments','corridor':'Corridor'})
        fig_rc.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',height=300,
            xaxis=dict(gridcolor='#dde3ea'),yaxis=dict(gridcolor='#dde3ea'),
            legend=dict(bgcolor='#ffffff'),margin=dict(t=10,b=10))
        st.plotly_chart(fig_rc,use_container_width=True)

    st.divider()
    c_c,c_d = st.columns(2)
    with c_c:
        st.markdown("**I-40 Terrain Profile**")
        i40_sub = df[df['corridor']=='I-40'].sort_values('segment_id')
        fig_e40 = go.Figure()
        fig_e40.add_trace(go.Scatter(
            x=list(range(len(i40_sub))),y=i40_sub['mean_elevation_m'],
            fill='tozeroy',line=dict(color='#64ffda',width=1.5),
            fillcolor='rgba(100,255,218,0.15)',name='Elevation'))
        fig_e40.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',height=250,
            xaxis=dict(title='Segment (W→E)',gridcolor='#dde3ea'),
            yaxis=dict(title='Elevation (m)',gridcolor='#dde3ea'),
            margin=dict(t=10,b=10))
        st.plotly_chart(fig_e40,use_container_width=True)

    with c_d:
        st.markdown("**I-75 Terrain Profile**")
        i75_sub = df[df['corridor']=='I-75'].sort_values('segment_id')
        fig_e75 = go.Figure()
        fig_e75.add_trace(go.Scatter(
            x=list(range(len(i75_sub))),y=i75_sub['mean_elevation_m'],
            fill='tozeroy',line=dict(color='#ff9f43',width=1.5),
            fillcolor='rgba(255,159,67,0.15)',name='Elevation'))
        fig_e75.update_layout(
            paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',
            font_color='#1a1a2e',height=250,
            xaxis=dict(title='Segment (N→S)',gridcolor='#dde3ea'),
            yaxis=dict(title='Elevation (m)',gridcolor='#dde3ea'),
            margin=dict(t=10,b=10))
        st.plotly_chart(fig_e75,use_container_width=True)

# ══ TAB 5: TABLE ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="section-header">Filtered Segments ({len(filtered):,} of {len(df):,})</div>',
                unsafe_allow_html=True)
    display_cols = ['segment_id','corridor','risk_score','risk_category',
                    'mean_slope_deg','elev_range_m','terrain_roughness',
                    'mean_disp_mm','mean_coherence','rock_class',
                    'AADT_2022','truck_pct']
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols].round(3),
        use_container_width=True,hide_index=True,
        column_config={
            'risk_score': st.column_config.ProgressColumn(
                'Risk Score',min_value=0,max_value=1,format='%.3f'),
            'AADT_2022': st.column_config.NumberColumn('AADT 2022',format='%d'),
        })
    c_dl1,c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button("⬇ Download Filtered CSV",
            filtered[display_cols].round(3).to_csv(index=False),
            "tdot_filtered.csv","text/csv")
    with c_dl2:
        st.download_button("⬇ Download Top 10 Risk",
            df.nlargest(10,'risk_score')[display_cols].round(3).to_csv(index=False),
            "top10_risk.csv","text/csv")

# ── Segment 149 deep dive figure inside tab6
with tab6:
    st.divider()
    st.markdown('<div class="section-header">Segment 149 — Full Deep Dive Figure</div>',
                unsafe_allow_html=True)
    from PIL import Image as PILImage
    img149 = PILImage.open('outputs/figures/fig12_segment149_deepdive.png')
    st.image(img149, use_container_width=True,
             caption='Segment 149 · 2-year InSAR time series · Coherence · '
                     'Corridor location · Key finding: InSAR detects what terrain misses')

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='color:#5a6a7a;font-size:0.8rem;text-align:center'>
TDOT Slope Movement Monitoring · I-40 + I-75 · East Tennessee · 432 Segments · 346 km<br>
Terrain: USGS 3DEP 30m · InSAR: Sentinel-1 ASF HyP3 · Geology: USGS · Traffic: FHWA HPMS 2022<br>
⚠ Proof of Concept — Labels pending TDOT USMP ground-truth validation
</p>
""", unsafe_allow_html=True)

# ══ SEGMENT 149 FIGURE — added to clustering tab
SEG149_FIG = 'outputs/figures/fig12_segment149_deepdive.png'
# ══ TAB 6: CLUSTERING ════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Unsupervised Risk Clustering — No Proxy Labels</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#5a6a7a;font-size:0.9rem'>
    K-Means clustering (K=3, silhouette=0.376) applied directly to real feature measurements —
    terrain, InSAR displacement, geology, and traffic. No proxy labels or assumptions.
    Clusters are ranked by composite terrain + InSAR signal to assign Low/Moderate/High risk.
    This approach is methodologically stronger than supervised classification with proxy labels.
    </p>
    """, unsafe_allow_html=True)

    from PIL import Image
    img_clust = Image.open('outputs/figures/fig11_unsupervised_clustering.png')
    st.image(img_clust, use_container_width=True,
             caption='K-Means clustering (K=3) · PCA visualization · '
                     'Feature heatmap · Silhouette score selection')

    st.divider()

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown('<div class="section-header">Cluster Profiles</div>',
                    unsafe_allow_html=True)
        cluster_info = pd.DataFrame({
            'Cluster': [0, 1, 2],
            'Risk':    ['Moderate','Low','High'],
            'N segs':  [149, 128, 65],
            'Mean Slope°': [5.2, 4.1, 13.7],
            'Elev Range m': [102, 79, 275],
            'Mean Disp mm': [2.91, 0.0, 0.0],
            'Trend mm/mo':  [-0.158, 0.0, 0.0],
            'InSAR':  ['Real data','No coverage','No coverage'],
        })
        st.dataframe(cluster_info, use_container_width=True, hide_index=True)
        st.markdown("""
        <p style='color:#5a6a7a;font-size:0.82rem;margin-top:8px'>
        ⚠ Clusters 1 and 2 show 0.0 displacement because they fall outside
        the Sentinel-1 InSAR swath coverage, not because they have zero movement.
        Terrain features drive their risk ranking.
        </p>
        """, unsafe_allow_html=True)

    with col_c2:
        st.markdown('<div class="section-header">Unsupervised vs Proxy Labels (I-40)</div>',
                    unsafe_allow_html=True)
        df_comp = pd.read_csv('data/processed/model_table/combined_corridors.csv')
        df_i40  = df_comp[df_comp['corridor']=='I-40'].copy()
        if 'risk_category_unsup' not in df_i40.columns:
            df_i40['risk_category_unsup'] = 'N/A'

        comp_data = pd.DataFrame({
            'Method': ['Proxy Labels (old)', 'Unsupervised K-Means (new)'],
            'Low':     [
                int((df_i40['risk_category']=='Low').sum()),
                int((df_i40['risk_category_unsup']=='Low').sum())
            ],
            'Moderate': [
                int((df_i40['risk_category']=='Moderate').sum()),
                int((df_i40['risk_category_unsup']=='Moderate').sum())
            ],
            'High': [
                int((df_i40['risk_category']=='High').sum()),
                int((df_i40['risk_category_unsup']=='High').sum())
            ],
        })
        st.dataframe(comp_data, use_container_width=True, hide_index=True)

        st.markdown("""
        <p style='color:#5a6a7a;font-size:0.82rem;margin-top:8px'>
        Unsupervised clustering produces 65 High-risk segments vs 66 from proxy labels —
        remarkably consistent. This validates that the terrain signal is strong enough
        to identify high-risk zones without any labeled training data.
        </p>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Segment 149 — Key Finding</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#1a1a2e;font-size:0.95rem'>
    Segment 149 lands in <b style='color:#4a90d9'>Moderate</b> cluster (risk score 0.584)
    driven entirely by its <b style='color:#ff9f43'>-1.785 mm/month</b> real InSAR subsidence trend
    over 2 years — despite having only 5.2° slope.
    <br><br>
    This demonstrates the value of InSAR: <b>a terrain-only model would classify this segment
    as Low risk</b> (flat ground). The satellite displacement signal elevates it to Moderate,
    flagging it for inspection. This is the core capability the TDOT USMP modernization needs.
    </p>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Segment", "149")
    col_s2.metric("Cluster", "0 (Moderate)")
    col_s3.metric("Trend", "-1.785 mm/mo", delta="Real Sentinel-1 signal")
    col_s4.metric("Slope", "5.2°", delta="Would be Low without InSAR")
