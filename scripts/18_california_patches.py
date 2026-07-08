"""
Generate overlapping inference patches (ZS mode) for the Dixie Fire
(California USA — 2021). Temperate conifer forest (Sierra Nevada).

STRIDE=128 (50% overlap). Output directly to Google Drive ZS folder.

Run from the project root (after 24_california_download.py):
    python -u scripts/25_california_patches.py
"""

import json, re, sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject as warp_reproject
from collections import defaultdict
from tqdm import tqdm

SITE           = "california"
CRS_UTM        = CRS.from_epsg(32610)   # UTM 10N — Sierra Nevada
DNBR_THRESHOLD = 0.20                   # Conifer forest: higher threshold
PATCH_SIZE     = 256
STRIDE         = 128
MAX_CLOUD_FRAC = 0.20                   # Sierra Nevada autumn: very strict

LOCAL_BASE = Path(__file__).parent.parent
PRE_DIR    = LOCAL_BASE / "data" / SITE / "raw_prefire"
POST_DIR   = LOCAL_BASE / "data" / SITE / "raw_postfire"

DRIVE_BASE = Path("G:/Mon Drive/GeoAI/wildfire-spread")
if not DRIVE_BASE.exists():
    print(f"WARNING: Drive not found, writing locally.")
    DRIVE_BASE = LOCAL_BASE

PATCH_OUT = DRIVE_BASE / "data" / "zs" / SITE / "patches"
MANIFEST  = DRIVE_BASE / "data" / "zs" / SITE / "manifest.json"
PATCH_OUT.mkdir(parents=True, exist_ok=True)


def tile_id_from_dir(d):
    m = re.search(r"_T(\d{2}[A-Z]{3})_", d.name)
    return f"T{m.group(1)}" if m else d.name[:10]

def load_band(jp2_path, dst_crs, dst_res=10.0):
    with rasterio.open(jp2_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_res)
        data = np.zeros((height, width), dtype=np.float32)
        warp_reproject(source=rasterio.band(src, 1), destination=data,
                       src_crs=src.crs, src_transform=src.transform,
                       dst_crs=dst_crs, dst_transform=transform,
                       resampling=Resampling.bilinear)
        return data, transform

def match_dims(arr, H, W):
    if arr.shape == (H, W): return arr
    out = np.zeros((H, W), dtype=arr.dtype)
    h, w = min(arr.shape[0], H), min(arr.shape[1], W)
    out[:h, :w] = arr[:h, :w]
    return out

def read_scene(item_dir, tile_id):
    date_dirs = sorted(item_dir.iterdir())
    if not date_dirs: return None
    dd = date_dirs[0]
    for b in ["B02","B03","B04","B08","B8A","B11","B12","SCL"]:
        if not (dd / f"{b}.jp2").exists():
            print(f"  SKIP {tile_id}: missing {b}.jp2"); return None
    b08, transform = load_band(dd / "B08.jp2", CRS_UTM)
    H, W = b08.shape
    bands = {"B08": b08}
    for b in ["B02","B03","B04"]:
        arr, _ = load_band(dd / f"{b}.jp2", CRS_UTM)
        bands[b] = match_dims(arr, H, W)
    for b in ["B8A","B11","B12"]:
        arr, _ = load_band(dd / f"{b}.jp2", CRS_UTM)
        bands[b] = match_dims(arr, H, W)
    scl, _ = load_band(dd / "SCL.jp2", CRS_UTM)
    scl = match_dims(scl.astype(np.uint8), H, W)
    return {"bands": bands, "scl": scl, "transform": transform, "H": H, "W": W}

def compute_dnbr(pre, post):
    H, W = post["H"], post["W"]
    def nbr(b8, b12):
        d = b8 + b12; d = np.where(d == 0, 1e-6, d); return (b8 - b12) / d
    return nbr(match_dims(pre["bands"]["B08"],H,W), match_dims(pre["bands"]["B12"],H,W)) \
         - nbr(post["bands"]["B08"], post["bands"]["B12"])

def cloud_mask(scl): return np.isin(scl, [3, 8, 9, 10])

def index_scenes(root_dir):
    by_tile = defaultdict(list)
    for d in root_dir.iterdir():
        if d.is_dir(): by_tile[tile_id_from_dir(d)].append(d)
    return by_tile


print("=" * 65)
print(f"Patch extraction — {SITE.upper()} (ZS mode, stride={STRIDE})")
print("=" * 65)

pre_by_tile  = index_scenes(PRE_DIR)
post_by_tile = index_scenes(POST_DIR)
common_tiles = sorted(set(pre_by_tile) & set(post_by_tile))
print(f"Tiles with both pre and post: {common_tiles}")
if not common_tiles: sys.exit("ERROR: No tiles with both pre and post scenes.")

manifest, total_patches = [], 0

for tile_id in common_tiles:
    print(f"\n[{tile_id}]")
    pre  = read_scene(pre_by_tile[tile_id][0],  tile_id)
    post = read_scene(post_by_tile[tile_id][0], tile_id)
    if pre is None or post is None: continue

    H, W = post["H"], post["W"]
    print(f"  Grid: {H} x {W} px")
    dnbr      = compute_dnbr(pre, post)
    fire_mask = (dnbr > DNBR_THRESHOLD).astype(np.uint8)
    cloud     = cloud_mask(post["scl"])
    print(f"  Fire pixels: {fire_mask.sum():,} ({100*fire_mask.sum()/(H*W):.1f}%)")

    b02,b03,b04 = post["bands"]["B02"],post["bands"]["B03"],post["bands"]["B04"]
    b08,b8a     = post["bands"]["B08"],post["bands"]["B8A"]
    b11,b12     = post["bands"]["B11"],post["bands"]["B12"]
    eps = 1e-6
    stack = np.stack([
        b02.astype(np.int16), b03.astype(np.int16), b04.astype(np.int16),
        b08.astype(np.int16), b8a.astype(np.int16), b11.astype(np.int16),
        b12.astype(np.int16),
        ((b08-b04)/(b08+b04+eps)*10000).astype(np.int16),
        ((b08-b12)/(b08+b12+eps)*10000).astype(np.int16),
        ((b03-b08)/(b03+b08+eps)*10000).astype(np.int16),
        fire_mask.astype(np.int16),
    ], axis=0)

    tile_patches = tile_fire = tile_cloud = 0
    for r in tqdm(range(0, H-PATCH_SIZE+1, STRIDE), desc=f"  {tile_id}", leave=False):
        for c in range(0, W-PATCH_SIZE+1, STRIDE):
            if cloud[r:r+PATCH_SIZE, c:c+PATCH_SIZE].mean() > MAX_CLOUD_FRAC:
                tile_cloud += 1; continue
            patch = stack[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            if (patch[:7] == 0).all(axis=0).mean() > 0.3: continue
            np.save(PATCH_OUT / f"{tile_id}_r{r:05d}_c{c:05d}.npy", patch)
            tile_patches += 1
            if fire_mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE].mean() > 0: tile_fire += 1

    print(f"  Saved: {tile_patches} patches (fire={tile_fire}, cloud-skip={tile_cloud})")
    total_patches += tile_patches
    manifest.append({"tile": tile_id, "patches": tile_patches, "fire_patches": tile_fire,
                     "H": H, "W": W, "transform": list(post["transform"]), "crs": str(CRS_UTM)})

with open(MANIFEST, "w") as f:
    json.dump({"site": SITE, "stride": STRIDE, "patch_size": PATCH_SIZE, "crs_epsg": 32610,
               "dnbr_thresh": DNBR_THRESHOLD, "total": total_patches, "tiles": manifest}, f, indent=2)

print(f"\n{'='*65}")
print(f"Done. Total patches: {total_patches}")
print("Next: python -u scripts/19_cerrado_download.py")
