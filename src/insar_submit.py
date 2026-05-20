import hyp3_sdk
import asf_search as asf
import datetime, json
from collections import defaultdict

print("Searching for Sentinel-1 scenes over I-40 corridor...")

AOI = 'POLYGON((-83.80 35.75,-82.50 35.75,-82.50 36.00,-83.80 36.00,-83.80 35.75))'

results = asf.search(
    platform         = asf.PLATFORM.SENTINEL1,
    processingLevel  = asf.PRODUCT_TYPE.SLC,
    beamMode         = 'IW',
    intersectsWith   = AOI,
    start            = datetime.datetime(2023, 1, 1),
    end              = datetime.datetime(2024, 6, 30),
    maxResults       = 50,
)

# Group by track
tracks = defaultdict(list)
for r in results:
    p = r.properties
    tracks[p['pathNumber']].append({
        'granule': p['sceneName'],
        'date':    p['startTime'][:10],
    })

best_track = max(tracks, key=lambda k: len(tracks[k]))
scenes = sorted(tracks[best_track], key=lambda x: x['date'])

# Deduplicate by date (keep first per date)
seen_dates = {}
for s in scenes:
    if s['date'] not in seen_dates:
        seen_dates[s['date']] = s
scenes_dedup = sorted(seen_dates.values(), key=lambda x: x['date'])
print(f"  Unique dates on track {best_track}: {len(scenes_dedup)}")

# Find 12-day pairs
pairs = []
for i in range(len(scenes_dedup)-1):
    d1 = datetime.datetime.strptime(scenes_dedup[i]['date'],   '%Y-%m-%d')
    d2 = datetime.datetime.strptime(scenes_dedup[i+1]['date'], '%Y-%m-%d')
    if 10 <= (d2-d1).days <= 14:
        pairs.append((scenes_dedup[i]['granule'], scenes_dedup[i+1]['granule']))

print(f"  Valid 12-day pairs: {len(pairs)}")

# Submit jobs
hyp3 = hyp3_sdk.HyP3('https://hyp3-api.asf.alaska.edu')
job_ids = []

print("\nSubmitting InSAR jobs...")
for ref, sec in pairs[:5]:
    batch = hyp3.submit_insar_job(
        granule1                  = ref,
        granule2                  = sec,
        name                      = f'i40_{ref[17:25]}',
        looks                     = '20x4',
        include_displacement_maps = True,
        apply_water_mask          = False,
    )
    # Batch is iterable — get first job
    for job in batch:
        job_ids.append(job.job_id)
        print(f"  Submitted job {job.job_id}: {ref[17:25]} → {sec[17:25]}")

with open('data/raw/sentinel1_insar/job_ids.json', 'w') as f:
    json.dump(job_ids, f, indent=2)

print(f"\n{len(job_ids)} jobs submitted.")
print("Processing takes 30-60 min. Run src/insar_download.py to check status.")
