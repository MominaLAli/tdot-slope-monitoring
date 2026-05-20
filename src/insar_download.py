import hyp3_sdk
import json, os, zipfile, requests
from pathlib import Path

hyp3 = hyp3_sdk.HyP3('https://hyp3-api.asf.alaska.edu')

with open('data/raw/sentinel1_insar/job_ids.json') as f:
    job_ids = json.load(f)

print(f"Checking status of {len(job_ids)} jobs...\n")

all_done = True
download_urls = []

for jid in job_ids:
    job = hyp3.get_job_by_id(jid)
    status = job.status_code
    name   = job.name
    print(f"  {name}  →  {status}")
    if status != 'SUCCEEDED':
        all_done = False
    else:
        for f in job.files:
            if f['filename'].endswith('.zip'):
                download_urls.append(f['url'])

if not all_done:
    print("\nNot all jobs done yet. Wait a few more minutes and re-run this script.")
else:
    print(f"\nAll jobs complete. Downloading {len(download_urls)} files...")
    out_dir = Path('data/raw/sentinel1_insar')

    for url in download_urls:
        fname = url.split('/')[-1].split('?')[0]
        fpath = out_dir / fname
        if fpath.exists():
            print(f"  Already exists: {fname}")
            continue
        print(f"  Downloading: {fname}")
        r = requests.get(url, stream=True)
        with open(fpath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        # Unzip
        with zipfile.ZipFile(fpath, 'r') as z:
            z.extractall(out_dir)
        print(f"  Extracted: {fname}")

    print("\nListing extracted files:")
    tifs = list(out_dir.rglob('*.tif'))
    for t in tifs:
        print(f"  {t.name}")
    print(f"\nDownload complete. {len(tifs)} GeoTIFF files ready.")
