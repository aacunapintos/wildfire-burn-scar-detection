"""
Overnight pipeline v2: download + patch extraction for 3 ZS inference sites.

    Chile      (Valparaiso 2023)       -- Mediterranean WUI
    California (Dixie Fire 2021)       -- Temperate conifer Sierra Nevada
    Cerrado    (Mato Grosso 2023)      -- Tropical dry savanna

Runs scripts 15-20 in sequence. Stops on first failure.
Patches land directly in G:/Mon Drive/GeoAI/wildfire-spread/data/zs/{site}/



Usage (from project root):
    python -u scripts/21_run_zs_pipeline.py 2>&1 | tee logs/zs_pipeline.log

Expected runtime: 3-6 hours (dominated by CDSE downloads ~30-50 GB total).
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    "scripts/15_chile_download.py",
    "scripts/16_chile_patches.py",
    "scripts/17_california_download.py",
    "scripts/18_california_patches.py",
    "scripts/19_cerrado_download.py",
    "scripts/20_cerrado_patches.py",
]

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

start_total = time.time()

for i, script in enumerate(SCRIPTS, 1):
    print()
    print("=" * 65)
    print(f"[{i}/{len(SCRIPTS)}] {script}")
    print("=" * 65)
    t0 = time.time()

    result = subprocess.run([sys.executable, "-u", script], check=False)

    elapsed = (time.time() - t0) / 60
    if result.returncode != 0:
        print(f"\nPIPELINE STOPPED — {script} failed (exit {result.returncode})")
        print(f"Elapsed: {(time.time()-start_total)/60:.1f} min")
        sys.exit(result.returncode)

    print(f"\n  Done in {elapsed:.1f} min")

total_min = (time.time() - start_total) / 60
print()
print("=" * 65)
print(f"All 6 scripts completed in {total_min:.1f} min ({total_min/60:.1f} h)")
print()
print("Patches ready at:")
print("  G:/Mon Drive/GeoAI/wildfire-spread/data/zs/chile/patches/")
print("  G:/Mon Drive/GeoAI/wildfire-spread/data/zs/california/patches/")
print("  G:/Mon Drive/GeoAI/wildfire-spread/data/zs/cerrado/patches/")
