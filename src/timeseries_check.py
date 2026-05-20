# Quick check — how many pairs do we currently have ready
from pathlib import Path
import pandas as pd, re

disp_files = sorted(Path('data/raw/sentinel1_insar').rglob('*_vert_disp.tif'))

def get_date(p):
    dates = re.findall(r'(\d{8})T', p.stem)
    return dates[0] if dates else 'unknown'

dates = sorted([get_date(f) for f in disp_files])
print(f"Displacement files ready: {len(disp_files)}")
print(f"Date range: {dates[0]} → {dates[-1]}")
print(f"\nAll dates:")
for d in dates:
    print(f"  {d[:4]}-{d[4:6]}-{d[6:]}")
