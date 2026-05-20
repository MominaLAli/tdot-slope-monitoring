import hyp3_sdk, json, os, zipfile, requests
from pathlib import Path

hyp3 = hyp3_sdk.HyP3('https://hyp3-api.asf.alaska.edu')

with open('data/raw/sentinel1_insar/job_ids_extended.json') as f:
    job_ids = json.load(f)

print(f"Checking {len(job_ids)} jobs...\n")

status_counts = {'SUCCEEDED':0,'RUNNING':0,'PENDING':0,'FAILED':0}
download_urls = []
failed = []

for jid in job_ids:
    job    = hyp3.get_job_by_id(jid)
    status = job.status_code
    status_counts[status] = status_counts.get(status, 0) + 1
    if status == 'SUCCEEDED':
        for f in job.files:
            if f['filename'].endswith('.zip'):
                download_urls.append((f['url'], job.name))
    elif status == 'FAILED':
        failed.append(job.name)

print(f"  SUCCEEDED: {status_counts['SUCCEEDED']}")
print(f"  RUNNING:   {status_counts.get('RUNNING',0)}")
print(f"  PENDING:   {status_counts.get('PENDING',0)}")
print(f"  FAILED:    {status_counts.get('FAILED',0)}")

if failed:
    print(f"\n  Failed jobs: {failed}")

all_done = status_counts['SUCCEEDED'] == len(job_ids)

if not all_done:
    done_pct = status_counts['SUCCEEDED']/len(job_ids)*100
    print(f"\n  {done_pct:.0f}% complete. Re-run this script in a few minutes.")
else:
    print(f"\nAll done! Downloading {len(download_urls)} files...")
    out_dir = Path('data/raw/sentinel1_insar')

    for url, name in download_urls:
        fname = url.split('/')[-1].split('?')[0]
        fpath = out_dir / fname
        if fpath.exists():
            print(f"  Already exists: {fname[:50]}")
            continue
        print(f"  Downloading: {name}")
        r = requests.get(url, stream=True)
        with open(fpath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(fpath, 'r') as z:
            z.extractall(out_dir)
        os.remove(fpath)

    disp_files = list(out_dir.rglob('*_vert_disp.tif'))
    print(f"\n  Total displacement files ready: {len(disp_files)}")
    print("  Run src/insar_timeseries.py to extract time series features.")
