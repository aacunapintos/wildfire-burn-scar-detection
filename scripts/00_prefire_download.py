"""
Download Sentinel-2 L2A pre-fire imagery for Corrientes (October–November 2021).

This script downloads the same tile footprints used for post-fire training
but 2 months before the fire season started. These images are the "pre-fire"
time step needed for T=2 temporal fusion (Improvement #6).

Run from the project root:
    python scripts/00_prefire_download.py

Credentials: read from wildfire-spread/.env
Output: data/sentinel2/raw_prefire/<item_id>/<date>/B02.jp2  ...
Estimated size: ~3-5 GB, ~3-5 hours on a home connection.
"""

import os
import re
import sys
import time
import requests
import pystac_client
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# ── Credentials ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
CDSE_USER     = os.getenv("CDSE_USER")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")

if not CDSE_USER or not CDSE_PASSWORD:
    sys.exit("ERROR: CDSE_USER / CDSE_PASSWORD missing in .env")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / "data" / "sentinel2" / "raw_prefire"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Search parameters ─────────────────────────────────────────────────────────
# Same bounding box as training set (Corrientes, Argentina)
BBOX = [-59.5, -29.0, -56.0, -26.5]

# Pre-fire window: before the December 2021 fire season
TIME_START = "2021-10-01T00:00:00Z"
TIME_END   = "2021-11-30T23:59:59Z"

# Same tile IDs used for post-fire data
TARGET_TILES = {"T21JVJ", "T21JUH", "T21JWJ", "T21JVL", "T21JWK"}

MAX_CLOUD = 30   # relaxed vs post-fire (austral spring, more cloud variability)
MAX_PER_TILE = 2  # best 2 scenes per tile

# Bands to download (same as notebook 01)
BAND_ASSETS = {
    "B02": "B02_10m",
    "B03": "B03_10m",
    "B04": "B04_10m",
    "B08": "B08_10m",
    "B8A": "B8A_20m",
    "B11": "B11_20m",
    "B12": "B12_20m",
    "SCL": "SCL_20m",
}


# ── CDSE helpers ──────────────────────────────────────────────────────────────
def get_cdse_token(user, password):
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token",
        data={"client_id": "cdse-public", "grant_type": "password",
              "username": user, "password": password},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def get_product_uuid(item):
    href = item.assets.get("Product", item.assets.get("product", None))
    if href is None:
        return None
    m = re.search(r"Products\(([^)]+)\)", href.href)
    return m.group(1) if m else None


def s3_to_odata(s3_url, uuid):
    path  = s3_url.replace("s3://eodata/", "")
    parts = path.split("/")
    safe_idx  = next(i for i, p in enumerate(parts) if p.endswith(".SAFE"))
    safe_name = parts[safe_idx]
    rest      = parts[safe_idx + 1:]
    nodes     = "/".join(f"Nodes({p})" for p in rest)
    return (
        f"https://download.dataspace.copernicus.eu"
        f"/odata/v1/Products({uuid})/Nodes({safe_name})/{nodes}/$value"
    )


def download_file(href, token, output_path, chunk_size=1024 * 1024):
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(href, headers=headers, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(output_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=output_path.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))


def tile_id_from_item(item):
    """Extract MGRS tile ID (e.g. T21JVJ) from item ID."""
    m = re.search(r"_T(\d{2}[A-Z]{3})_", item.id)
    return f"T{m.group(1)}" if m else None


# ── Search catalog ────────────────────────────────────────────────────────────
print(f"Searching CDSE STAC for pre-fire imagery...")
print(f"  BBOX   : {BBOX}")
print(f"  Period : {TIME_START[:10]} → {TIME_END[:10]}")
print(f"  Tiles  : {sorted(TARGET_TILES)}")
print(f"  Cloud  : < {MAX_CLOUD}%")
print()

catalog = pystac_client.Client.open(
    "https://catalogue.dataspace.copernicus.eu/stac"
)

search  = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=BBOX,
    datetime=f"{TIME_START}/{TIME_END}",
    max_items=200,
)
all_items = list(search.items())
print(f"Total items returned: {len(all_items)}")

# Filter: cloud cover + target tiles only
from collections import defaultdict
by_tile = defaultdict(list)
for item in all_items:
    cloud = item.properties.get("eo:cloud_cover", 100)
    tile  = tile_id_from_item(item)
    if tile in TARGET_TILES and cloud < MAX_CLOUD:
        by_tile[tile].append(item)

# Sort per tile by cloud cover, keep best MAX_PER_TILE
selected = []
print("Selected scenes:")
for tile in sorted(TARGET_TILES):
    items_sorted = sorted(by_tile[tile],
                          key=lambda x: x.properties.get("eo:cloud_cover", 100))
    top = items_sorted[:MAX_PER_TILE]
    for it in top:
        date  = it.datetime.strftime("%Y-%m-%d")
        cloud = it.properties.get("eo:cloud_cover", 0)
        print(f"  {tile}  {date}  cloud={cloud:.1f}%  {it.id[:50]}")
    selected.extend(top)

if not selected:
    sys.exit(
        "\nNo scenes found. Try:\n"
        "  - Increasing MAX_CLOUD (currently {MAX_CLOUD}%)\n"
        "  - Widening the date range\n"
        "  - Checking CDSE catalog status"
    )

total_size_est = len(selected) * len(BAND_ASSETS) * 60  # ~60 MB per band
print(f"\n{len(selected)} scenes to download (~{total_size_est/1000:.1f} GB estimated)")
print()

# ── Download ──────────────────────────────────────────────────────────────────
token      = get_cdse_token(CDSE_USER, CDSE_PASSWORD)
token_time = time.time()
downloaded = []
skipped    = []
errors     = []

for item in selected:
    # Refresh token every 9 minutes
    if time.time() - token_time > 540:
        token      = get_cdse_token(CDSE_USER, CDSE_PASSWORD)
        token_time = time.time()
        print("  [token refreshed]")

    uuid      = get_product_uuid(item)
    date_str  = item.datetime.strftime("%Y%m%d")
    tile_dir  = OUT_DIR / item.id / date_str
    tile_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{item.id[:55]}")
    print(f"  Date: {date_str}  Cloud: {item.properties.get('eo:cloud_cover', 0):.1f}%")

    if uuid is None:
        print("  SKIP — could not extract product UUID")
        skipped.append(item.id)
        continue

    for band, asset_key in BAND_ASSETS.items():
        if asset_key not in item.assets:
            print(f"  SKIP  {band} (asset '{asset_key}' not found)")
            skipped.append(f"{item.id}/{band}")
            continue

        output_path = tile_dir / f"{band}.jp2"
        if output_path.exists() and output_path.stat().st_size > 100_000:
            print(f"  SKIP  {band} (already downloaded)")
            continue

        odata_url = s3_to_odata(item.assets[asset_key].href, uuid)
        try:
            download_file(odata_url, token, output_path)
            size_mb = output_path.stat().st_size / 1e6
            print(f"  OK    {band}  {size_mb:.1f} MB")
            downloaded.append(output_path)
        except Exception as e:
            print(f"  ERROR {band}: {e}")
            if output_path.exists():
                output_path.unlink()
            errors.append(f"{item.id}/{band}: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Download complete.")
print(f"  Files downloaded : {len(downloaded)}")
print(f"  Skipped          : {len(skipped)}")
print(f"  Errors           : {len(errors)}")
if errors:
    print("  Error details:")
    for e in errors[:5]:
        print(f"    {e}")
total_gb = sum(p.stat().st_size for p in OUT_DIR.rglob("*.jp2")) / 1e9
print(f"  Total on disk    : {total_gb:.2f} GB")
print(f"  Output dir       : {OUT_DIR.resolve()}")
print()
print("Next step:")
print("  python scripts/03b_paired_patches.py")
