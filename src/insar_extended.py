import hyp3_sdk
import asf_search as asf
import datetime, json
from collections import defaultdict

print("Searching for 2-year Sentinel-1 archive over I-40 corridor...")

AOI = 'POLYGON((-83.80 35.75,-82.50 35.75,-82.50 36.00,-83.80 36.00,-83.80 35.75))'

results = asf.search(
    platform        = asf.PLATFORM.SENTINEL1,
    processingLevel = asf.PRODUCT_TYPE.SLC,
    beamMode        = 'IW',
    intersectsWith  = AOI,
    start           = datetime.datetime(2022, 1, 1),
    end             = datetime.datetime(2024, 1, 1),
    maxResults      = 200,
)
print(f"  Total scenes found: {len(results)}")

# Group by track, deduplicate by date
tracks = defaultdict(dict)
for r in results:
    p = r.properties
    track = p['pathNumber']
    date  = p['startTime'][:10]
    if date not in tracks[track]:
        tracks[track][date] = p['sceneName']

best_track = max(tracks, key=lambda k: len(tracks[k]))
scenes = sorted([{'date': d, 'granule': g}
                 for d, g in tracks[best_track].items()],
                key=lambda x: x['date'])

print(f"  Track {best_track}: {len(scenes)} unique dates (2022–2024)")

# Find all 12-day pairs
pairs = []
for i in range(len(scenes)-1):
    d1 = datetime.datetime.strptime(scenes[i]['date'],   '%Y-%m-%d')
    d2 = datetime.datetime.strptime(scenes[i+1]['date'], '%Y-%m-%d')
    if 10 <= (d2-d1).days <= 14:
        pairs.append((scenes[i]['granule'], scenes[i+1]['granule'],
                      scenes[i]['date'],    scenes[i+1]['date']))

print(f"  Valid 12-day pairs: {len(pairs)}")

# Skip already submitted pairs (from job_ids.json)
with open('data/raw/sentinel1_insar/job_ids.json') as f:
    existing_ids = json.load(f)

# Submit new pairs (up to 30 — covers ~1 year)
hyp3 = hyp3_sdk.HyP3('https://hyp3-api.asf.alaska.edu')
credits = hyp3.check_credits()
print(f"  Available credits: {credits}")

new_ids  = []
to_submit = pairs[:30]  # 30 pairs = ~1 year coverage

print(f"\nSubmitting {len(to_submit)} new InSAR jobs...")
for ref, sec, d1, d2 in to_submit:
    try:
        batch = hyp3.submit_insar_job(
            granule1                  = ref,
            granule2                  = sec,
            name                      = f'i40ext_{d1}',
            looks                     = '20x4',
            include_displacement_maps = True,
            apply_water_mask          = False,
        )
        for job in batch:
            new_ids.append(job.job_id)
            print(f"  Submitted: {d1} → {d2}  (id: {job.job_id[:8]}...)")
    except Exception as e:
        print(f"  Failed {d1}: {e}")

# Save all job IDs
all_ids = existing_ids + new_ids
with open('data/raw/sentinel1_insar/job_ids_extended.json', 'w') as f:
    json.dump(all_ids, f, indent=2)

print(f"\n  New jobs submitted: {len(new_ids)}")
print(f"  Total jobs (including original 5): {len(all_ids)}")
print(f"  Remaining credits: {hyp3.check_credits()}")
print("\nJobs running on ASF servers. Check status later with:")
print("  python src/insar_download_extended.py")
