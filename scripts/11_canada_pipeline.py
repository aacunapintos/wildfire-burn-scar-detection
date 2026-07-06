"""
End-to-end pipeline: Download + Patches for North Slave Complex wildfire
(Northwest Territories, Canada — August 2023, boreal forest, ~163,000 ha burned).

Steps:
  1. Search and download Sentinel-2 L2A from CDSE (pre-fire Jun-Jul, post-fire Sep-Oct 2023)
  2. Reproject JP2 tiles to UTM 11N GeoTIFF
  3. Compute dNBR and extract 256x256 patches

Run with unbuffered output:
    python -u scripts/11_canada_pipeline.py

Monitor from a second terminal:
    Get-Content D:\\GeoAI\\wildfire-spread\\data\\canada_overnight.log -Wait -Tail 40
"""

import os, re, sys, json, time, site
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── timestamp helper ──────────────────────────────────────────────────────────
def ts(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── PROJ fix — must run before rasterio ──────────────────────────────────────
_proj_set = False
for _sp in site.getsitepackages():
    _proj = Path(_sp) / "rasterio" / "proj_data"
    if _proj.exists():
        os.environ["PROJ_DATA"] = str(_proj)
        os.environ["PROJ_LIB"]  = str(_proj)
        _proj_set = True
        break
if not _proj_set:
    _conda = Path(sys.executable).parent.parent / "Library" / "share" / "proj"
    if _conda.exists():
        os.environ["PROJ_DATA"] = str(_conda)
        os.environ["PROJ_LIB"]  = str(_conda)

import requests
import pystac_client
import numpy as np
import rasterio
import rasterio.warp
from rasterio.crs import CRS
from tqdm import tqdm
from dotenv import load_dotenv

# ── Credentials ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
CDSE_USER     = os.getenv("CDSE_USER")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")
if not CDSE_USER or not CDSE_PASSWORD:
    sys.exit("ERROR: CDSE_USER / CDSE_PASSWORD missing in .env")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent.parent
PRE_OUT_DIR   = BASE / "data" / "canada" / "raw_prefire"
POST_OUT_DIR  = BASE / "data" / "canada" / "raw_postfire"
PROC_PRE_DIR  = BASE / "data" / "canada" / "processed" / "prefire"
PROC_POST_DIR = BASE / "data" / "canada" / "processed" / "postfire"
PATCH_IMG_DIR  = BASE / "data" / "canada" / "patches" / "images"
PATCH_MASK_DIR = BASE / "data" / "canada" / "patches" / "masks_dnbr"

for d in (PRE_OUT_DIR, POST_OUT_DIR, PROC_PRE_DIR, PROC_POST_DIR,
          PATCH_IMG_DIR, PATCH_MASK_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Download parameters ───────────────────────────────────────────────────────
BBOX          = [-116.5, 61.8, -113.5, 63.2]
PRE_START     = "2023-06-01T00:00:00Z"
PRE_END       = "2023-07-31T23:59:59Z"
POST_START    = "2023-09-01T00:00:00Z"
POST_END      = "2023-10-15T23:59:59Z"
MAX_CLOUD_PRE  = 30
MAX_CLOUD_POST = 25
MAX_PER_TILE   = 2

BAND_ASSETS = {
    "B02": "B02_10m", "B03": "B03_10m", "B04": "B04_10m",
    "B08": "B08_10m", "B8A": "B8A_20m", "B11": "B11_20m",
    "B12": "B12_20m", "SCL": "SCL_20m",
}

# ── Patch parameters ──────────────────────────────────────────────────────────
CRS_UTM        = CRS.from_epsg(32611)   # UTM Zone 11N — Yellowknife / NWT
BANDS          = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
SCL_CLEAR      = {4, 5, 6}
PATCH_SIZE     = 256
DNBR_THRESHOLD = 0.10
MAX_CLOUD_FRAC = 0.25
FIRE_RATIO     = 0.60
MIN_FIRE_FRAC  = 0.05
RESOLUTION_10M = 10
RESOLUTION_20M = 20


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def get_cdse_token(user, password):
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token",
        data={"client_id": "cdse-public", "grant_type": "password",
              "username": user, "password": password}, timeout=30)
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
    return (f"https://download.dataspace.copernicus.eu"
            f"/odata/v1/Products({uuid})/Nodes({safe_name})/{nodes}/$value")


def download_file(href, token, output_path, chunk_size=1024*1024):
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
    print(f"\nSelected {label} scenes ({len(by_tile)} tiles):", flush=True)
    for tile in sorted(by_tile):
        top = sorted(by_tile[tile],
                     key=lambda x: x.properties.get("eo:cloud_cover", 100))[:max_per_tile]
        for it in top:
            date  = it.datetime.strftime("%Y-%m-%d")
            cloud = it.properties.get("eo:cloud_cover", 0)
            print(f"  {tile}  {date}  cloud={cloud:.1f}%  {it.id[:55]}", flush=True)
        selected.extend(top)
    return selected


def download_scenes(scenes, out_dir, token_state, label):
    token      = token_state["token"]
    token_time = token_state["time"]
    downloaded, skipped, errors = [], [], []
    total = len(scenes)

    for i, item in enumerate(scenes, 1):
        if time.time() - token_time > 540:
            token = get_cdse_token(CDSE_USER, CDSE_PASSWORD)
            token_time = time.time()
            token_state.update(token=token, time=token_time)
            ts("[token refreshed]")

        uuid     = get_product_uuid(item)
        date_str = item.datetime.strftime("%Y%m%d")
        item_dir = out_dir / item.id / date_str
        item_dir.mkdir(parents=True, exist_ok=True)

        cloud = item.properties.get("eo:cloud_cover", 0)
        ts(f"[{label}] scene {i}/{total}  {date_str}  cloud={cloud:.1f}%")

        if uuid is None:
            print(f"  SKIP — no UUID", flush=True)
            skipped.append(item.id)
            continue

        for band, asset_key in BAND_ASSETS.items():
            if asset_key not in item.assets:
                skipped.append(f"{item.id}/{band}")
                continue
            out_path = item_dir / f"{band}.jp2"
            if out_path.exists() and out_path.stat().st_size > 100_000:
                print(f"  SKIP  {band} (already exists)", flush=True)
                continue
            odata_url = s3_to_odata(item.assets[asset_key].href, uuid)
            try:
                download_file(odata_url, token, out_path)
                size_mb = out_path.stat().st_size / 1e6
                print(f"  OK    {band}  {size_mb:.1f} MB", flush=True)
                downloaded.append(out_path)
            except Exception as e:
                print(f"  ERROR {band}: {e}", flush=True)
                if out_path.exists():
                    out_path.unlink()
                errors.append(f"{item.id}/{band}: {e}")

    return downloaded, skipped, errors


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PATCHES
# ═══════════════════════════════════════════════════════════════════════════════

def reproject_band(src_path, dst_crs, resolution=10):
    with rasterio.open(src_path) as src:
        src_crs = src.crs if src.crs else CRS.from_epsg(32611)
        transform, width, height = rasterio.warp.calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=resolution)
        data = np.zeros((height, width), dtype=src.dtypes[0])
        rasterio.warp.reproject(
            source=rasterio.band(src, 1), destination=data,
            src_transform=src.transform, src_crs=src_crs,
            dst_transform=transform, dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.bilinear)
    return data, transform


def compute_index(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(a + b > 0, (a - b) / (a + b), 0.0).astype(np.float32)


def process_scene(scene_dir, out_dir, label=""):
    item_id  = scene_dir.parent.name
    date_str = scene_dir.name
    stem     = f"{item_id[-20:]}_{date_str}"
    out_path = out_dir / f"{stem}.tif"
    if out_path.exists():
        print(f"  SKIP {out_path.name} (already processed)", flush=True)
        return out_path
    scl_path = scene_dir / "SCL.jp2"
    if not scl_path.exists():
        print(f"  SKIP {stem} — SCL.jp2 missing", flush=True)
        return None
    scl_data, scl_transform = reproject_band(scl_path, CRS_UTM, RESOLUTION_10M)
    clear_mask = np.isin(scl_data, list(SCL_CLEAR))
    shape = scl_data.shape
    band_data = {}
    for band in BANDS:
        res  = RESOLUTION_20M if band in ("B8A", "B11", "B12") else RESOLUTION_10M
        jp2  = scene_dir / f"{band}.jp2"
        if not jp2.exists():
            band_data[band] = np.zeros(shape, dtype=np.int16)
            continue
        data, _ = reproject_band(jp2, CRS_UTM, RESOLUTION_10M)
        if data.shape != shape:
            aligned = np.zeros(shape, dtype=data.dtype)
            h, w = min(data.shape[0], shape[0]), min(data.shape[1], shape[1])
            aligned[:h, :w] = data[:h, :w]
            data = aligned
        band_data[band] = data
    B08  = band_data["B08"].astype(np.float32)
    NDVI = compute_index(B08, band_data["B04"])
    NBR  = compute_index(B08, band_data["B12"])
    NDWI = compute_index(band_data["B03"], B08)
    stack = np.stack([
        band_data["B02"], band_data["B03"], band_data["B04"],
        band_data["B08"], band_data["B8A"], band_data["B11"], band_data["B12"],
        (NDVI * 10000).astype(np.int16), (NBR  * 10000).astype(np.int16),
        (NDWI * 10000).astype(np.int16), clear_mask.astype(np.uint8),
    ], axis=0)
    profile = {"driver": "GTiff", "dtype": "int16", "compress": "lzw",
               "crs": CRS_UTM, "transform": scl_transform,
               "width": shape[1], "height": shape[0],
               "count": stack.shape[0], "nodata": -9999}
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)
        dst.update_tags(band_names="B02,B03,B04,B08,B8A,B11,B12,NDVI,NBR,NDWI,MASK",
                        date=date_str, item_id=item_id, phase=label)
    pct_clear = clear_mask.mean() * 100
    size_mb   = out_path.stat().st_size / 1e6
    print(f"  OK  {out_path.name}  ({size_mb:.0f} MB, {pct_clear:.1f}% clear) [{label}]",
          flush=True)
    return out_path


def collect_scenes(raw_dir):
    scenes = []
    for item_dir in sorted(raw_dir.iterdir()):
        if not item_dir.is_dir():
            continue
        for date_dir in sorted(item_dir.iterdir()):
            if date_dir.is_dir() and list(date_dir.glob("*.jp2")):
                scenes.append(date_dir)
    return scenes


def tile_id_from_name(name):
    m = re.search(r"_T(\d{2}[A-Z]{3})_", name)
    return f"T{m.group(1)}" if m else None


def best_tif_for_tile(tifs_by_tile, tile_id):
    candidates = tifs_by_tile.get(tile_id, [])
    if not candidates:
        return None
    def clear_frac(p):
        with rasterio.open(p) as src:
            return src.read(11).mean()
    return max(candidates, key=clear_frac)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

ts("=" * 65)
ts("Canada boreal forest pipeline — North Slave Complex, NWT 2023")
ts("=" * 65)
ts(f"BBOX : {BBOX}")
ts(f"Pre  : {PRE_START[:10]} to {PRE_END[:10]}")
ts(f"Post : {POST_START[:10]} to {POST_END[:10]}")
ts("")

# ── Step 1: Search ────────────────────────────────────────────────────────────
ts("Searching CDSE catalog...")
catalog = pystac_client.Client.open(
    "https://catalogue.dataspace.copernicus.eu/stac")

pre_items = list(catalog.search(
    collections=["sentinel-2-l2a"], bbox=BBOX,
    datetime=f"{PRE_START}/{PRE_END}", max_items=200).items())
ts(f"Pre-fire  items found : {len(pre_items)}")

post_items = list(catalog.search(
    collections=["sentinel-2-l2a"], bbox=BBOX,
    datetime=f"{POST_START}/{POST_END}", max_items=200).items())
ts(f"Post-fire items found : {len(post_items)}")

pre_selected  = select_scenes(pre_items,  MAX_CLOUD_PRE,  MAX_PER_TILE, "pre-fire")
post_selected = select_scenes(post_items, MAX_CLOUD_POST, MAX_PER_TILE, "post-fire")

if not pre_selected:
    sys.exit("ERROR: No pre-fire scenes — try increasing MAX_CLOUD_PRE.")
if not post_selected:
    sys.exit("ERROR: No post-fire scenes — try increasing MAX_CLOUD_POST.")

total_scenes = len(pre_selected) + len(post_selected)
est_gb = total_scenes * len(BAND_ASSETS) * 0.06
ts(f"Scenes to download : {total_scenes}  (~{est_gb:.1f} GB estimated)")
ts("")

# ── Step 2: Download ──────────────────────────────────────────────────────────
ts("Starting download...")
token_state = {"token": get_cdse_token(CDSE_USER, CDSE_PASSWORD), "time": time.time()}

dl_pre,  sk_pre,  err_pre  = download_scenes(pre_selected,  PRE_OUT_DIR,  token_state, "PRE")
dl_post, sk_post, err_post = download_scenes(post_selected, POST_OUT_DIR, token_state, "POST")

all_errors = err_pre + err_post
total_gb   = (sum(p.stat().st_size for p in PRE_OUT_DIR.rglob("*.jp2")) +
              sum(p.stat().st_size for p in POST_OUT_DIR.rglob("*.jp2"))) / 1e9

ts("")
ts("=" * 65)
ts("DOWNLOAD COMPLETE")
ts(f"  Files   : {len(dl_pre)+len(dl_post)}")
ts(f"  Errors  : {len(all_errors)}")
ts(f"  On disk : {total_gb:.2f} GB")
ts("=" * 65)
ts("")

if len(all_errors) > 0:
    for e in all_errors[:5]:
        print(f"  {e}", flush=True)
    sys.exit(f"Aborted: {len(all_errors)} download errors. Fix before running patches.")

# ── Step 3: JP2 → GeoTIFF ────────────────────────────────────────────────────
ts("Phase A — Reprojecting JP2 tiles to GeoTIFF (UTM 11N)...")

pre_scenes  = collect_scenes(PRE_OUT_DIR)
post_scenes = collect_scenes(POST_OUT_DIR)

pre_tifs_by_tile  = defaultdict(list)
post_tifs_by_tile = defaultdict(list)

ts(f"Pre-fire  scenes : {len(pre_scenes)}")
for scene_dir in pre_scenes:
    tif = process_scene(scene_dir, PROC_PRE_DIR, "PRE")
    if tif:
        tile = tile_id_from_name(scene_dir.parent.name)
        if tile:
            pre_tifs_by_tile[tile].append(tif)

ts(f"Post-fire scenes : {len(post_scenes)}")
for scene_dir in post_scenes:
    tif = process_scene(scene_dir, PROC_POST_DIR, "POST")
    if tif:
        tile = tile_id_from_name(scene_dir.parent.name)
        if tile:
            post_tifs_by_tile[tile].append(tif)

common_tiles = set(pre_tifs_by_tile) & set(post_tifs_by_tile)
ts(f"Common tiles (pre+post): {sorted(common_tiles)}")

if not common_tiles:
    sys.exit("ERROR: No tiles with both pre and post GeoTIFFs.")

# ── Step 4: dNBR + patches ────────────────────────────────────────────────────
ts("")
ts("Phase B — Computing dNBR and extracting 256x256 patches...")

rng           = np.random.default_rng(seed=42)
total_written = total_fire = total_bg = total_skip_cloud = 0
log_entries   = []

for tile in sorted(common_tiles):
    pre_tif  = best_tif_for_tile(pre_tifs_by_tile,  tile)
    post_tif = best_tif_for_tile(post_tifs_by_tile, tile)
    ts(f"Tile {tile}: {post_tif.name}")

    with rasterio.open(post_tif) as post_src, rasterio.open(pre_tif) as pre_src:
        H, W = post_src.height, post_src.width
        b08_post  = post_src.read(4).astype(np.float32)
        b12_post  = post_src.read(7).astype(np.float32)
        mask_post = post_src.read(11)

        if pre_src.crs != post_src.crs or pre_src.transform != post_src.transform:
            b08_pre = np.zeros((H, W), dtype=np.float32)
            b12_pre = np.zeros((H, W), dtype=np.float32)
            for src_band, dst_arr in [(4, b08_pre), (7, b12_pre)]:
                rasterio.warp.reproject(
                    source=rasterio.band(pre_src, src_band),
                    destination=dst_arr,
                    src_transform=pre_src.transform, src_crs=pre_src.crs,
                    dst_transform=post_src.transform, dst_crs=post_src.crs,
                    resampling=rasterio.warp.Resampling.bilinear)
        else:
            b08_pre = pre_src.read(4).astype(np.float32)
            b12_pre = pre_src.read(7).astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            nbr_pre  = np.where(b08_pre  + b12_pre  > 0,
                                (b08_pre  - b12_pre)  / (b08_pre  + b12_pre  + 1e-6), 0.)
            nbr_post = np.where(b08_post + b12_post > 0,
                                (b08_post - b12_post) / (b08_post + b12_post + 1e-6), 0.)

        dnbr      = (nbr_pre - nbr_post).astype(np.float32)
        burn_mask = (dnbr > DNBR_THRESHOLD).astype(np.uint8)
        burned_pct = burn_mask.mean() * 100
        print(f"  dNBR [{dnbr.min():.3f}, {dnbr.max():.3f}]  "
              f"burned {burned_pct:.1f}% of tile", flush=True)

        log_entries.append({"tile": tile, "pre_tif": pre_tif.name,
                             "post_tif": post_tif.name,
                             "dnbr_min": float(dnbr.min()),
                             "dnbr_max": float(dnbr.max()),
                             "burned_pct": float(burned_pct)})

        rows = range(0, H - PATCH_SIZE + 1, PATCH_SIZE)
        cols = range(0, W - PATCH_SIZE + 1, PATCH_SIZE)
        fire_positions, bg_positions = [], []

        for r in rows:
            for c in cols:
                pm    = burn_mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                pc    = mask_post[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                if 1 - pc.mean() > MAX_CLOUD_FRAC:
                    total_skip_cloud += 1
                    continue
                if pm.mean() >= MIN_FIRE_FRAC:
                    fire_positions.append((r, c))
                else:
                    bg_positions.append((r, c))

        n_fire = len(fire_positions)
        n_bg   = min(int(n_fire * (1 - FIRE_RATIO) / FIRE_RATIO), len(bg_positions))
        sampled_bg = (rng.choice(len(bg_positions), size=n_bg, replace=False).tolist()
                      if n_bg > 0 and bg_positions else [])
        selected = fire_positions + [bg_positions[i] for i in sampled_bg]
        print(f"  Fire={n_fire}  BG={n_bg}  Total={len(selected)}", flush=True)

        all_bands   = post_src.read()
        stem_prefix = f"{post_tif.stem[:25]}_{tile}"

        for r, c in tqdm(selected, desc=f"  Patches {tile}", leave=True, file=sys.stdout):
            img_patch  = all_bands[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE].astype(np.int16)
            mask_patch = burn_mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE].astype(np.float32)
            name = f"{stem_prefix}_r{r:05d}_c{c:05d}.npy"
            np.save(PATCH_IMG_DIR  / name, img_patch)
            np.save(PATCH_MASK_DIR / name, mask_patch)
            total_written += 1
            if mask_patch.mean() >= MIN_FIRE_FRAC:
                total_fire += 1
            else:
                total_bg += 1

# ── Manifest ──────────────────────────────────────────────────────────────────
manifest = {"event": "North Slave Complex wildfire, NWT, Canada — August 2023",
            "dnbr_threshold": DNBR_THRESHOLD, "patch_size": PATCH_SIZE,
            "total_patches": total_written, "fire_patches": total_fire,
            "background_patches": total_bg, "tiles": log_entries}
with open(BASE / "data" / "canada" / "patches" / "manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)

ts("")
ts("=" * 65)
ts("PIPELINE COMPLETE")
ts(f"  Patches total : {total_written}")
ts(f"    Fire        : {total_fire}  ({total_fire/max(total_written,1)*100:.1f}%)")
ts(f"    Background  : {total_bg}")
ts(f"  Cloud skipped : {total_skip_cloud}")
ts("=" * 65)
