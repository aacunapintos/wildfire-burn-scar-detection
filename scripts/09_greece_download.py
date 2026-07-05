"""
Download Sentinel-2 L2A pre-fire and post-fire imagery for the Alexandroupolis
wildfire (Dadia-Lefkimi-Soufli Forest, Evros, Greece — August 2023).

This event burned ~810 km2 of Mediterranean shrubland, making it the largest
wildfire recorded in the EU. It is used as the out-of-distribution test site
for geographic generalization (cross-biome zero-shot evaluation).

Run from the project root:
    python scripts/09_greece_download.py

Credentials: read from wildfire-spread/.env  (CDSE_USER, CDSE_PASSWORD)
Output:
    data/greece/raw_prefire/<item_id>/<date>/B02.jp2  ...   (May–Jul 2023)
    data/greece/raw_postfire/<item_id>/<date>/B02.jp2 ...   (Sep–Oct 2023)

Estimated download size: ~5–8 GB, ~4–6 hours on a home connection.
Next step: python scripts/10_greece_patches.py
"""

import os
import re
import sys
import time
import requests
import pystac_client
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm

# ── Credentials ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
CDSE_USER     = os.getenv("CDSE_USER")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")

if not CDSE_USER or not CDSE_PASSWORD:
    sys.exit("ERROR: CDSE_USER / CDSE_PASSWORD missing in .env")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = Path(__file__).parent.parent
PRE_OUT_DIR    = BASE / "data" / "greece" / "raw_prefire"
POST_OUT_DIR   = BASE / "data" / "greece" / "raw_postfire"
PRE_OUT_DIR.mkdir(parents=True, exist_ok=True)
POST_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Search parameters ─────────────────────────────────────────────────────────
# Evros / Dadia-Lefkimi area, NE Greece.  Fire center: ~41.1N, 26.3E.
BBOX = [25.6, 40.6, 27.4, 42.0]

# Pre-fire: green season before the August 2023 fire
PRE_START  = "2023-05-01T00:00:00Z"
PRE_END    = "2023-07-31T23:59:59Z"

# Post-fire: first clear-sky pass after fire containment (late August / Sep–Oct)
POST_START = "2023-09-01T00:00:00Z"
POST_END   = "2023-10-31T23:59:59Z"

MAX_CLOUD_PRE  = 20   # relaxed for pre-fire (spring, variable cloud)
MAX_CLOUD_POST = 15   # strict for post-fire (clear burn scar needed)
MAX_PER_TILE   = 2    # best N scenes per MGRS tile

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
    asset = item.assets.get("Product") or item.assets.get("product")
    if asset is None:
        return None
    m = re.search(r"Products\(([^)]+)\)", asset.href)
    return m.group(1) if m else None


def s3_to_odata(s3_url, uuid):
    path      = s3_url.replace("s3://eodata/", "")
    parts     = path.split("/")
    safe_idx  = next(i for i, p in enumerate(parts) if p.endswith(".SAFE"))
    safe_name = parts[safe_idx]
    nodes     = "/".join(f"Nodes({p})" for p in parts[safe_idx + 1:])
    return (
        f"https://download.dataspace.copernicus.eu"
        f"/odata/v1/Products({uuid})/Nodes({safe_name})/{nodes}/$value"
    )


def download_file(href, token, output_path, chunk_size=1024 * 1024):
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(href, headers=headers, stream=True, timeout=300) as r:
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
    m = re.search(r"_T(\d{2}[A-Z]{3})_", item.id)
    return f"T{m.group(1)}" if m else item.id[:10]


def select_scenes(items, max_cloud, max_per_tile, label):
    by_tile = defaultdict(list)
    for item in items:
        cloud = item.properties.get("eo:cloud_cover", 100)
        tile  = tile_id_from_item(item)
        if cloud < max_cloud:
            by_tile[tile].append(item)

    selected = []
    print(f"\nSelected {label} scenes:")
    for tile in sorted(by_tile):
        top = sorted(by_tile[tile],
                     key=lambda x: x.properties.get("eo:cloud_cover", 100))[:max_per_tile]
        for it in top:
            date  = it.datetime.strftime("%Y-%m-%d")
            cloud = it.properties.get("eo:cloud_cover", 0)
            print(f"  {tile}  {date}  cloud={cloud:.1f}%  {it.id[:55]}")
        selected.extend(top)
    return selected


def download_scenes(scenes, out_dir, token_state, label):
    token      = token_state["token"]
    token_time = token_state["time"]
    downloaded, skipped, errors = [], [], []

    for item in scenes:
        if time.time() - token_time > 540:
            token = get_cdse_token(CDSE_USER, CDSE_PASSWORD)
            token_time = time.time()
            token_state.update(token=token, time=token_time)
            print("  [token refreshed]")

        uuid     = get_product_uuid(item)
        date_str = item.datetime.strftime("%Y%m%d")
        item_dir = out_dir / item.id / date_str
        item_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{label}] {item.id[:55]}")
        print(f"  Date: {date_str}  Cloud: {item.properties.get('eo:cloud_cover', 0):.1f}%")

        if uuid is None:
            print("  SKIP — could not extract product UUID")
            skipped.append(item.id)
            continue

        for band, asset_key in BAND_ASSETS.items():
            if asset_key not in item.assets:
                print(f"  SKIP  {band} (asset '{asset_key}' not in item)")
                skipped.append(f"{item.id}/{band}")
                continue

            out_path = item_dir / f"{band}.jp2"
            if out_path.exists() and out_path.stat().st_size > 100_000:
                print(f"  SKIP  {band} (already downloaded)")
                continue

            odata_url = s3_to_odata(item.assets[asset_key].href, uuid)
            try:
                download_file(odata_url, token, out_path)
                size_mb = out_path.stat().st_size / 1e6
                print(f"  OK    {band}  {size_mb:.1f} MB")
                downloaded.append(out_path)
            except Exception as e:
                print(f"  ERROR {band}: {e}")
                if out_path.exists():
                    out_path.unlink()
                errors.append(f"{item.id}/{band}: {e}")

    return downloaded, skipped, errors


# ── Search ────────────────────────────────────────────────────────────────────
catalog = pystac_client.Client.open(
    "https://catalogue.dataspace.copernicus.eu/stac"
)

print("=" * 65)
print("Searching CDSE — Alexandroupolis / Dadia-Lefkimi wildfire 2023")
print("=" * 65)
print(f"  BBOX : {BBOX}")
print(f"  Pre  : {PRE_START[:10]} to {PRE_END[:10]}  (cloud < {MAX_CLOUD_PRE}%)")
print(f"  Post : {POST_START[:10]} to {POST_END[:10]}  (cloud < {MAX_CLOUD_POST}%)")

pre_items = list(catalog.search(
    collections=["sentinel-2-l2a"], bbox=BBOX,
    datetime=f"{PRE_START}/{PRE_END}", max_items=200,
).items())
print(f"\nPre-fire items returned by STAC  : {len(pre_items)}")

post_items = list(catalog.search(
    collections=["sentinel-2-l2a"], bbox=BBOX,
    datetime=f"{POST_START}/{POST_END}", max_items=200,
).items())
print(f"Post-fire items returned by STAC : {len(post_items)}")

pre_selected  = select_scenes(pre_items,  MAX_CLOUD_PRE,  MAX_PER_TILE, "pre-fire")
post_selected = select_scenes(post_items, MAX_CLOUD_POST, MAX_PER_TILE, "post-fire")

if not pre_selected:
    sys.exit(
        "\nERROR: No pre-fire scenes found. "
        "Try increasing MAX_CLOUD_PRE or widening the date range."
    )
if not post_selected:
    sys.exit(
        "\nERROR: No post-fire scenes found. "
        "Try increasing MAX_CLOUD_POST or widening the date range."
    )

total_scenes = len(pre_selected) + len(post_selected)
est_gb = total_scenes * len(BAND_ASSETS) * 0.06  # ~60 MB per band per scene
print(f"\nTotal scenes to download : {total_scenes} ({est_gb:.1f} GB estimated)")
print()

# ── Download ──────────────────────────────────────────────────────────────────
token_state = {"token": get_cdse_token(CDSE_USER, CDSE_PASSWORD),
               "time": time.time()}

dl_pre,  sk_pre,  err_pre  = download_scenes(pre_selected,  PRE_OUT_DIR,  token_state, "PRE")
dl_post, sk_post, err_post = download_scenes(post_selected, POST_OUT_DIR, token_state, "POST")

# ── Summary ───────────────────────────────────────────────────────────────────
all_errors = err_pre + err_post
total_gb   = (
    sum(p.stat().st_size for p in PRE_OUT_DIR.rglob("*.jp2"))
  + sum(p.stat().st_size for p in POST_OUT_DIR.rglob("*.jp2"))
) / 1e9

print()
print("=" * 65)
print("Download complete.")
print(f"  Files downloaded  : {len(dl_pre) + len(dl_post)}")
print(f"  Skipped           : {len(sk_pre) + len(sk_post)}")
print(f"  Errors            : {len(all_errors)}")
if all_errors:
    print("  Error details (first 5):")
    for e in all_errors[:5]:
        print(f"    {e}")
print(f"  Total on disk     : {total_gb:.2f} GB")
print(f"  Pre-fire  dir     : {PRE_OUT_DIR}")
print(f"  Post-fire dir     : {POST_OUT_DIR}")
print()
print("Next step:")
print("  python scripts/10_greece_patches.py")
