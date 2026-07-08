"""
Download Sentinel-2 L2A pre-fire and post-fire imagery for the Valparaiso wildfires
(Chile — February 2023).

Mediterranean WUI (wildland-urban interface). Worst fire in Chilean history:
15,000 homes destroyed, 131 deaths. Focused on the Lago Penelas - Villa Alemana
corridor where the scar is dense and spectrally clear.
Used as zero-shot inference site (not training).

Run from the project root:
    python -u scripts/22_chile_download.py

Credentials: read from wildfire-spread/.env  (CDSE_USER, CDSE_PASSWORD)
Output:
    data/chile/raw_prefire/<item_id>/<date>/B02.jp2  ...
    data/chile/raw_postfire/<item_id>/<date>/B02.jp2 ...

Next step: python -u scripts/16_chile_patches.py
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

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
CDSE_USER     = os.getenv("CDSE_USER")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")

if not CDSE_USER or not CDSE_PASSWORD:
    sys.exit("ERROR: CDSE_USER / CDSE_PASSWORD missing in .env")

BASE         = Path(__file__).parent.parent
PRE_OUT_DIR  = BASE / "data" / "chile" / "raw_prefire"
POST_OUT_DIR = BASE / "data" / "chile" / "raw_postfire"
PRE_OUT_DIR.mkdir(parents=True, exist_ok=True)
POST_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lago Penelas - Villa Alemana - Quilpue corridor (main scar)
BBOX = [-71.8, -33.2, -71.2, -32.8]

# Pre-fire: Chilean summer before the fire (reference vegetation state)
PRE_START  = "2022-11-01T00:00:00Z"
PRE_END    = "2023-01-15T23:59:59Z"

# Post-fire: late February onward — fire contained Feb 8, clear skies guaranteed
POST_START = "2023-02-15T00:00:00Z"
POST_END   = "2023-04-30T23:59:59Z"

MAX_CLOUD_PRE  = 10   # Chilean central valley summer: extremely clear
MAX_CLOUD_POST = 15
MAX_PER_TILE   = 2

BAND_ASSETS = {
    "B02": "B02_10m", "B03": "B03_10m", "B04": "B04_10m", "B08": "B08_10m",
    "B8A": "B8A_20m", "B11": "B11_20m", "B12": "B12_20m", "SCL": "SCL_20m",
}


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
    path     = s3_url.replace("s3://eodata/", "")
    parts    = path.split("/")
    safe_idx = next(i for i, p in enumerate(parts) if p.endswith(".SAFE"))
    nodes    = "/".join(f"Nodes({p})" for p in parts[safe_idx + 1:])
    return (
        f"https://download.dataspace.copernicus.eu"
        f"/odata/v1/Products({uuid})/Nodes({parts[safe_idx]})/{nodes}/$value"
    )


def download_file(href, token, output_path, chunk_size=1024 * 1024):
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(href, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(output_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=output_path.name, leave=False
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
        if item.properties.get("eo:cloud_cover", 100) < max_cloud:
            by_tile[tile_id_from_item(item)].append(item)
    selected = []
    print(f"\nSelected {label} scenes:")
    for tile in sorted(by_tile):
        top = sorted(by_tile[tile],
                     key=lambda x: x.properties.get("eo:cloud_cover", 100))[:max_per_tile]
        for it in top:
            print(f"  {tile}  {it.datetime.strftime('%Y-%m-%d')}  "
                  f"cloud={it.properties.get('eo:cloud_cover', 0):.1f}%")
        selected.extend(top)
    return selected


def download_scenes(scenes, out_dir, token_state, label):
    token, token_time = token_state["token"], token_state["time"]
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
        print(f"\n[{label}] {item.id[:55]}  {date_str}")
        if uuid is None:
            print("  SKIP — no UUID"); skipped.append(item.id); continue
        for band, asset_key in BAND_ASSETS.items():
            if asset_key not in item.assets:
                skipped.append(f"{item.id}/{band}"); continue
            out_path = item_dir / f"{band}.jp2"
            if out_path.exists() and out_path.stat().st_size > 100_000:
                print(f"  SKIP  {band}"); continue
            try:
                download_file(s3_to_odata(item.assets[asset_key].href, uuid), token, out_path)
                print(f"  OK    {band}  {out_path.stat().st_size/1e6:.1f} MB")
                downloaded.append(out_path)
            except Exception as e:
                print(f"  ERROR {band}: {e}")
                if out_path.exists(): out_path.unlink()
                errors.append(f"{item.id}/{band}: {e}")
    return downloaded, skipped, errors


catalog = pystac_client.Client.open("https://catalogue.dataspace.copernicus.eu/stac")

print("=" * 65)
print("Searching CDSE — Valparaiso wildfire, Chile 2023")
print("=" * 65)
print(f"  BBOX : {BBOX}")

pre_items  = list(catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX,
    datetime=f"{PRE_START}/{PRE_END}", max_items=200).items())
post_items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX,
    datetime=f"{POST_START}/{POST_END}", max_items=200).items())
print(f"  Pre STAC items: {len(pre_items)}  |  Post STAC items: {len(post_items)}")

pre_sel  = select_scenes(pre_items,  MAX_CLOUD_PRE,  MAX_PER_TILE, "pre-fire")
post_sel = select_scenes(post_items, MAX_CLOUD_POST, MAX_PER_TILE, "post-fire")

if not pre_sel:  sys.exit("\nERROR: No pre-fire scenes. Increase MAX_CLOUD_PRE.")
if not post_sel: sys.exit("\nERROR: No post-fire scenes. Increase MAX_CLOUD_POST.")

print(f"\nScenes to download: {len(pre_sel)+len(post_sel)}")
token_state = {"token": get_cdse_token(CDSE_USER, CDSE_PASSWORD), "time": time.time()}

dl_pre,  sk_pre,  err_pre  = download_scenes(pre_sel,  PRE_OUT_DIR,  token_state, "PRE")
dl_post, sk_post, err_post = download_scenes(post_sel, POST_OUT_DIR, token_state, "POST")

all_errors = err_pre + err_post
total_gb   = (sum(p.stat().st_size for p in PRE_OUT_DIR.rglob("*.jp2"))
            + sum(p.stat().st_size for p in POST_OUT_DIR.rglob("*.jp2"))) / 1e9

print(f"\n{'='*65}")
print(f"Done. Files: {len(dl_pre)+len(dl_post)}  Errors: {len(all_errors)}  "
      f"Disk: {total_gb:.2f} GB")
if all_errors:
    for e in all_errors[:5]: print(f"  {e}")
    sys.exit(f"COMPLETED WITH {len(all_errors)} ERRORS")
print("Next: python -u scripts/16_chile_patches.py")
