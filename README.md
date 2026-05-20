# TDOT Slope Movement Monitoring System

## East Tennessee Highway Corridor Risk Assessment
### Proof of Concept — I-40 + I-75 Mountain Corridors

---

## Project Overview

A GIS/ML pipeline integrating real satellite InSAR displacement data, USGS terrain analysis, state geology maps, and federal traffic counts to identify and prioritize slope instability risk along East Tennessee highways. Built toward TDOT Unstable Slope Management Program (USMP) modernization.

---

## Corridors Covered

| Corridor | Section | Length | Segments |
|----------|---------|--------|----------|
| I-40 | Cookeville to NC Border | 274 km | 342 |
| I-75 | Jellico Mountain Section | 72 km | 90 |
| **Total** | | **346 km** | **432** |

---

## Data Sources (All Real, All Free)

| Layer | Source | Resolution | Notes |
|-------|--------|------------|-------|
| Terrain/DEM | USGS 3DEP | 30m | TNM API download |
| InSAR Displacement | Sentinel-1, ASF HyP3 | ~80m | 35 pairs Jan 2022 – Jan 2024 |
| InSAR Coherence | Sentinel-1, ASF HyP3 | ~80m | Real satellite signal |
| Geology | USGS State Geology Map TN | Polygon | Rock type per segment |
| Traffic | FHWA HPMS 2022 | Segment | AADT + truck percentage |
| Labels | Terrain + InSAR composite | — | Proxy — pending USMP data |

---

## Features Extracted Per Segment

### Terrain (USGS 3DEP DEM)

| Feature | Description |
|---------|-------------|
| `mean_slope_deg` | Average slope angle in degrees |
| `max_slope_deg` | Maximum slope in 500m buffer |
| `std_slope_deg` | Slope variability |
| `mean_elevation_m` | Average elevation |
| `elev_range_m` | Elevation relief in buffer |
| `mean_curvature` | Terrain convexity/concavity |
| `terrain_roughness` | Standard deviation of elevation in 5x5 window |
| `stream_proximity_pct` | Percent pixels in lowest 10th percentile elevation |

### InSAR (Sentinel-1 ASF HyP3 — 35 pairs, 2 years)

| Feature | Description |
|---------|-------------|
| `mean_disp_mm` | Mean vertical displacement in mm |
| `max_disp_mm` | Maximum displacement value |
| `disp_std_mm` | Displacement variability across pairs |
| `disp_trend_mm_per_month` | Linear displacement trend over 2 years |
| `mean_coherence` | Mean InSAR coherence (0 = noise, 1 = perfect) |
| `n_pairs` | Number of valid pairs used |

### Geology (USGS State Geology Map TN)

| Feature | Description |
|---------|-------------|
| `rock_class` | carbonate / clastic_sedimentary / metamorphic_crystalline / unconsolidated / other |
| `rock_risk_score` | 1–4 failure susceptibility scale |

### Traffic (FHWA HPMS 2022)

| Feature | Description |
|---------|-------------|
| `AADT_2022` | Annual Average Daily Traffic |
| `truck_pct` | Percentage of truck traffic |
| `truck_AADT` | Truck volume (AADT × truck_pct) |

---

## Risk Score Formula

```
Risk Score =
    0.35 × ML probability (Random Forest)
  + 0.25 × Terrain susceptibility
  + 0.20 × InSAR movement signal (displacement + trend)
  + 0.12 × Road exposure (AADT + trucks)
  + 0.08 × Geology factor
```

### Terrain Susceptibility

```
terrain_susc = 0.4 × norm(mean_slope_deg)
             + 0.3 × norm(elev_range_m)
             + 0.3 × norm(terrain_roughness)
```

### InSAR Signal

```
insar_signal = 0.4 × norm(|mean_disp_mm|)
             + 0.4 × norm(|disp_trend_mm_per_month|)
             + 0.2 × norm(disp_std_mm)
```

### Road Exposure

```
road_exposure = 0.6 × norm(AADT_2022)
              + 0.4 × norm(truck_AADT)
```

---

## Risk Category Thresholds

| Category | Score Range | I-40 | I-75 | Total |
|----------|-------------|------|------|-------|
| Low | 0.00 – 0.25 | 242 | 53 | 295 |
| Moderate | 0.25 – 0.50 | 34 | 10 | 44 |
| High | 0.50 – 0.75 | 66 | 25 | 91 |
| Critical | 0.75 – 1.00 | 0 | 2 | 2 |

---

## ML Model Performance

| Model | CV ROC-AUC | Segments |
|-------|------------|----------|
| Random Forest — I-40 only | 0.952 ± 0.055 | 342 |
| Random Forest — Combined | 0.945 ± 0.050 | 432 |
| Gradient Boosting | 0.986 ± 0.006 | 342 |
| Logistic Regression | 0.997 ± 0.002 | 342 |

> **Note:** High AUC reflects proxy labels derived from same features. Will recalibrate with real USMP ground-truth labels when available.

---

## Key Findings

1. **I-75 segments 1023–1024 (Jellico)** are the only Critical-rated segments — clastic sedimentary rock, 15°+ slopes, confirmed by negative displacement signal. Geologically consistent with known Jellico area slope concerns.

2. **Segment 149 (I-40)** shows **-1.785 mm/month** sustained subsidence over 2 years of real Sentinel-1 data — flagged as High risk (score 0.526) despite only 5.2° slope. InSAR catches what terrain-only inspection would miss.

3. **Clastic sedimentary rock** dominates high-risk segments on both corridors — consistent with Appalachian landslide literature (sandstone/shale failure planes).

4. **Mean coherence improved from 0.431 (5 pairs) to 0.546 (35 pairs)** — confirming that longer temporal stacks significantly improve signal quality in forested Appalachian terrain.

5. **I-40 has higher AADT** (62,590 vs 34,851) but **I-75 Jellico shows higher terrain-driven risk per segment** — different risk profiles requiring different management strategies.

---

## Project Structure

```
tdot_slope_monitoring/
├── data/
│   ├── raw/
│   │   ├── roads/                    # Corridor geometries (UTM + WGS84)
│   │   ├── dem_lidar/                # USGS 3DEP DEM tiles + clipped rasters
│   │   ├── sentinel1_insar/          # 35 ASF HyP3 InSAR GeoTIFF pairs
│   │   ├── geology/                  # USGS TN geology shapefile + units CSV
│   │   └── traffic/                  # FHWA HPMS AADT per segment
│   └── processed/
│       ├── corridor/                 # Segment GeoJSONs (I-40 + I-75)
│       ├── terrain_features/         # DEM-derived feature CSVs
│       ├── insar_features/           # InSAR features + 2-year timeseries
│       └── model_table/              # Combined feature tables
├── src/
│   ├── corridor.py                   # I-40 geometry + segmentation
│   ├── corridor_i75.py               # I-75 geometry + segmentation
│   ├── terrain.py                    # DEM download (USGS TNM API)
│   ├── features.py                   # Terrain feature extraction (I-40)
│   ├── features_fixed.py             # Terrain with corrected geotransform
│   ├── features_i75.py               # Terrain feature extraction (I-75)
│   ├── geology.py                    # USGS geology spatial join
│   ├── traffic.py                    # FHWA AADT assignment
│   ├── insar_submit.py               # ASF HyP3 — initial 5 job submission
│   ├── insar_download.py             # Job status check + download
│   ├── insar_extended.py             # ASF HyP3 — 30 extended job submission
│   ├── insar_download_extended.py    # Extended job status + download
│   ├── insar_features.py             # InSAR zonal extraction (5 pairs)
│   ├── insar_timeseries.py           # 2-year time series + trend per segment
│   ├── model_real.py                 # I-40 ML model (real InSAR)
│   ├── model_final.py                # I-40 ML model (all 4 data sources)
│   └── combine_corridors.py          # Multi-corridor combined model
├── outputs/
│   ├── maps/                         # GeoJSON risk outputs
│   ├── figures/                      # Static map exports
│   └── tables/
│       ├── top_risk_segments.csv     # Top 10 highest-risk segments
│       ├── feature_importance.csv    # Random Forest feature importance
│       └── project_summary.csv       # Full project statistics
├── streamlit_app.py                  # Interactive 5-tab dashboard
└── README.md                         # This file
```

---

## How to Run

```bash
# 1. Activate environment
conda activate tdot_slope

# 2. Navigate to project
cd ~/Desktop/tdot_slope_monitoring

# 3. Launch dashboard
streamlit run streamlit_app.py
```

### Rebuild Pipeline from Scratch

```bash
# Corridor geometry
python src/corridor.py
python src/corridor_i75.py

# Terrain features
python src/terrain.py
python src/features_fixed.py
python src/features_i75.py

# Geology and traffic
python src/geology.py
python src/traffic.py

# InSAR (requires NASA Earthdata account)
python src/insar_submit.py
python src/insar_download.py
python src/insar_extended.py
python src/insar_download_extended.py
python src/insar_features.py
python src/insar_timeseries.py

# Model
python src/model_final.py
python src/combine_corridors.py

# Dashboard
streamlit run streamlit_app.py
```

---

## Requirements

```bash
conda create -n tdot_slope python=3.11 -y
conda activate tdot_slope
conda install -c conda-forge richdem -y
pip install geopandas rasterio rioxarray shapely pyproj folium \
    matplotlib pandas numpy scikit-learn xgboost lightgbm \
    streamlit plotly requests osmnx scipy hyp3_sdk asf_search \
    streamlit-folium
```

### NASA Earthdata Account Required

InSAR processing via ASF HyP3 requires a free NASA Earthdata account:

1. Register at: `https://urs.earthdata.nasa.gov/users/new`
2. Store credentials in `~/.netrc`:

```
machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
```

---

## Dashboard Features

| Tab | Content |
|-----|---------|
| 🗺 Risk Map | Interactive dark-theme map with both corridors, risk color-coding, hover tooltips |
| 📊 Analysis | Risk score profiles, feature importance, slope vs displacement, risk vs AADT |
| 🪨 Geology & InSAR | Risk by rock type, coherence distribution, displacement boxplots |
| 🛣 Corridor Compare | I-40 vs I-75 side-by-side metrics, risk breakdown, terrain profiles |
| 📋 Segment Table | Filterable table of all 432 segments with CSV download |

### Sidebar Filters

- Corridor selection (I-40 / I-75)
- Risk category (Low / Moderate / High / Critical)
- Mean slope range
- Rock class
- Minimum AADT

---

## What Is Needed for Production Deployment

| Item | Status | Next Step |
|------|--------|-----------|
| TDOT USMP ground-truth points | Missing | Request from Dr. Li / TDOT |
| Real AADT from TDOT GIS | Partial | TDOT GIS portal (API was down) |
| High-resolution LiDAR | Missing | Tennessee LiDAR portal |
| Spatial cross-validation | Pending | Implement after USMP labels |
| Additional corridors | Pending | I-81, US-441, US-64 |
| Live InSAR pipeline | Pending | ASF API automation |
| ArcGIS integration | Pending | ArcPy export layer |

### When USMP Data Becomes Available

```bash
# Drop USMP shapefile into:
data/raw/usmp_points/usmp_points.shp

# Then run label assignment + retrain:
python src/label_from_usmp.py
python src/model_final.py
python src/combine_corridors.py
streamlit run streamlit_app.py
```

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total segments | 432 |
| Corridors | 2 (I-40 + I-75) |
| Total corridor length | 346 km |
| InSAR pairs (I-40) | 35 |
| InSAR temporal baseline | 730 days (2 years) |
| Critical risk segments | 2 (I-75 Jellico) |
| High risk segments | 91 |
| Max subsidence rate | -1.785 mm/month (Segment 149) |
| Mean coherence | 0.546 |
| CV ROC-AUC (combined) | 0.945 ± 0.050 |
| Data sources | 4 (all real, all free) |


