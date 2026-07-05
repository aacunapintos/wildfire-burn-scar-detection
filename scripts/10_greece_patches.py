"""
Preprocessing pipeline for Greece 2023 Alexandroupolis wildfire data.

Reads:
    data/greece/raw_prefire/<item_id>/<date>/*.jp2    (downloaded by 09_greece_download.py)
    data/greece/raw_postfire/<item_id>/<date>/*.jp2

Writes:
    data/greece/processed/prefire/<stem>.tif           (reprojected GeoTIFFs)
    data/greece/processed/postfire/<stem>.tif
    data/greece/patches/images/<name>.npy              (11, 256, 256) post-fire bands
    data/greece/patches/masks_dnbr/<name>.npy          (256, 256) binary burn mask

Patch format matches Corrientes/Córdoba pipelines:
    band order: B02, B03, B04, B08, B8A, B11, B12, NDVI, NBR, NDWI, MASK  (int16, DN×10000)
    mask: 1 = burned (dNBR > DNBR_THRESHOLD), 0 = unburned

dNBR = NBR_pre − NBR_post   where   NBR = (B08 − B12) / (B08 + B12)
A positive dNBR indicates burned area. The Alexandroupolis fire burned
Mediterranean shrubland/forest (Pinus brutia, Cistus), a biome completely
different from the Corrientes subtropical savanna used for training.

Run from the project root:
    python scripts/10_greece_patches.py

Requirements: rasterio, numpy, tqdm  (same conda env as other scripts)
Estimated time: 30–90 min on local CPU (depends on tile count and size).
"""

import os
import re
import sys
import json
import site
from pathlib import Path

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

import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows
from rasterio.crs import CRS
from tqdm import tqdm
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = Path(__file__).parent.parent
RAW_PRE_DIR    = BASE / "data" / "greece" / "raw_prefire"
RAW_POST_DIR   = BASE / "data" / "greece" / "raw_postfire"
PROC_DIR       = BASE / "data" / "greece" / "processed"
PROC_PRE_DIR   = PROC_DIR / "prefire"
PROC_POST_DIR  = PROC_DIR / "postfire"
PATCH_IMG_DIR  = BASE / "data" / "greece" / "patches" / "images"
PATCH_MASK_DIR = BASE / "data" / "greece" / "patches" / "masks_dnbr"

for d in (PROC_PRE_DIR, PROC_POST_DIR, PATCH_IMG_DIR, PATCH_MASK_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# UTM zone 35N covers lon 24–30°E — correct for NE Greece / Evros region
CRS_UTM        = CRS.from_epsg(32635)
BANDS          = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
SCL_CLEAR      = {4, 5, 6}        # SCL classes: vegetation, bare soil, water
PATCH_SIZE     = 256
DNBR_THRESHOLD = 0.10             # minimum dNBR for burned pixel (low-to-moderate)
MAX_CLOUD_FRAC = 0.25             # max cloud fraction in a patch
FIRE_RATIO     = 0.60             # fraction of patches to sample from fire areas
MIN_FIRE_FRAC  = 0.05             # minimum fire fraction for a patch to be "fire"

RESOLUTION_10M = 10
RESOLUTION_20M = 20


# ── Phase A: helpers ──────────────────────────────────────────────────────────
def reproject_band(src_path, dst_crs, resolution=10):
    with rasterio.open(src_path) as src:
        src_crs = src.crs if src.crs else CRS.from_epsg(32635)
        transform, width, height = rasterio.warp.calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=resolution,
        )
        data = np.zeros((height, width), dtype=src.dtypes[0])
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.bilinear,
        )
    return data, transform


def compute_index(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a - b) / denom, 0.0).astype(np.float32)


def process_scene(scene_dir, out_dir, label=""):
    item_id  = scene_dir.parent.name
    date_str = scene_dir.name
    stem     = f"{item_id[-20:]}_{date_str}"
    out_path = out_dir / f"{stem}.tif"

    if out_path.exists():
        print(f"  SKIP {out_path.name} (already processed)")
        return out_path

    scl_path = scene_dir / "SCL.jp2"
    if not scl_path.exists():
        print(f"  SKIP {stem} [{label}] — SCL.jp2 missing")
        return None

    scl_data, scl_transform = reproject_band(scl_path, CRS_UTM, RESOLUTION_10M)
    clear_mask = np.isin(scl_data, list(SCL_CLEAR))
    shape = scl_data.shape

    band_data = {}
    for band in BANDS:
        res   = RESOLUTION_20M if band in ("B8A", "B11", "B12") else RESOLUTION_10M
        jp2   = scene_dir / f"{band}.jp2"
        if not jp2.exists():
            print(f"  WARN {band}.jp2 missing in {stem} — filling zeros")
            band_data[band] = np.zeros(shape, dtype=np.int16)
            continue
        data, _ = reproject_band(jp2, CRS_UTM, RESOLUTION_10M)
        if data.shape != shape:
            aligned = np.zeros(shape, dtype=data.dtype)
            h = min(data.shape[0], shape[0])
            w = min(data.shape[1], shape[1])
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
        (NDVI * 10000).astype(np.int16),
        (NBR  * 10000).astype(np.int16),
        (NDWI * 10000).astype(np.int16),
        clear_mask.astype(np.uint8),
    ], axis=0)

    profile = {
        "driver": "GTiff", "dtype": "int16", "compress": "lzw",
        "crs": CRS_UTM, "transform": scl_transform,
        "width": shape[1], "height": shape[0],
        "count": stack.shape[0], "nodata": -9999,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)
        dst.update_tags(
            band_names="B02,B03,B04,B08,B8A,B11,B12,NDVI,NBR,NDWI,MASK",
            date=date_str, item_id=item_id, phase=label,
        )

    pct_clear = clear_mask.mean() * 100
    size_mb   = out_path.stat().st_size / 1e6
    print(f"  OK  {out_path.name}  ({size_mb:.0f} MB, {pct_clear:.1f}% clear) [{label}]")
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
    """Return the GeoTIFF path with highest clear-pixel fraction for a tile."""
    candidates = tifs_by_tile.get(tile_id, [])
    if not candidates:
        return None
    def clear_frac(p):
        with rasterio.open(p) as src:
            mask = src.read(11)   # band 11 = MASK (0-indexed)
            return mask.mean()
    return max(candidates, key=clear_frac)


# ── Phase A: process raw JP2 tiles → reprojected GeoTIFFs ────────────────────
print("=" * 65)
print("Phase A — Converting raw JP2 tiles to reprojected GeoTIFFs")
print("=" * 65)

pre_scenes  = collect_scenes(RAW_PRE_DIR)
post_scenes = collect_scenes(RAW_POST_DIR)

if not pre_scenes:
    sys.exit(
        f"ERROR: No pre-fire JP2 tiles found in {RAW_PRE_DIR}\n"
        "Run scripts/09_greece_download.py first."
    )
if not post_scenes:
    sys.exit(
        f"ERROR: No post-fire JP2 tiles found in {RAW_POST_DIR}\n"
        "Run scripts/09_greece_download.py first."
    )

print(f"Pre-fire scenes  : {len(pre_scenes)}")
print(f"Post-fire scenes : {len(post_scenes)}")

# Index processed tifs by tile ID
pre_tifs_by_tile  = defaultdict(list)
post_tifs_by_tile = defaultdict(list)

print("\nProcessing pre-fire:")
for scene_dir in pre_scenes:
    tif = process_scene(scene_dir, PROC_PRE_DIR, label="PRE")
    if tif:
        tile = tile_id_from_name(scene_dir.parent.name)
        if tile:
            pre_tifs_by_tile[tile].append(tif)

print("\nProcessing post-fire:")
for scene_dir in post_scenes:
    tif = process_scene(scene_dir, PROC_POST_DIR, label="POST")
    if tif:
        tile = tile_id_from_name(scene_dir.parent.name)
        if tile:
            post_tifs_by_tile[tile].append(tif)

pre_tiles  = set(pre_tifs_by_tile)
post_tiles = set(post_tifs_by_tile)
common_tiles = pre_tiles & post_tiles

print(f"\nPre-fire tiles   : {sorted(pre_tiles)}")
print(f"Post-fire tiles  : {sorted(post_tiles)}")
print(f"Common tiles     : {sorted(common_tiles)}")

if not common_tiles:
    sys.exit(
        "ERROR: No tiles have both pre-fire and post-fire GeoTIFFs.\n"
        "Check MGRS tile IDs in the downloaded data."
    )


# ── Phase B: compute dNBR and extract patches ─────────────────────────────────
print()
print("=" * 65)
print("Phase B — Computing dNBR and extracting 256x256 patches")
print("=" * 65)

rng = np.random.default_rng(seed=42)

total_written = 0
total_fire    = 0
total_bg      = 0
total_skip_cloud  = 0
total_skip_nopair = 0

log_entries = []   # for documentation

for tile in sorted(common_tiles):
    pre_tif  = best_tif_for_tile(pre_tifs_by_tile,  tile)
    post_tif = best_tif_for_tile(post_tifs_by_tile, tile)

    print(f"\nTile {tile}")
    print(f"  Pre-fire  : {pre_tif.name}")
    print(f"  Post-fire : {post_tif.name}")

    with rasterio.open(post_tif) as post_src, \
         rasterio.open(pre_tif)  as pre_src:

        H, W = post_src.height, post_src.width

        # Read NBR bands for dNBR computation
        # Band indices (1-indexed in rasterio): B08=4, B12=7, MASK=11
        b08_post = post_src.read(4).astype(np.float32)
        b12_post = post_src.read(7).astype(np.float32)
        mask_post = post_src.read(11)   # SCL clear mask

        # Pre-fire at same spatial extent (reproject if needed)
        if pre_src.crs != post_src.crs or pre_src.transform != post_src.transform:
            b08_pre_full  = np.zeros((H, W), dtype=np.float32)
            b12_pre_full  = np.zeros((H, W), dtype=np.float32)
            rasterio.warp.reproject(
                source=rasterio.band(pre_src, 4),
                destination=b08_pre_full,
                src_transform=pre_src.transform,
                src_crs=pre_src.crs,
                dst_transform=post_src.transform,
                dst_crs=post_src.crs,
                resampling=rasterio.warp.Resampling.bilinear,
            )
            rasterio.warp.reproject(
                source=rasterio.band(pre_src, 7),
                destination=b12_pre_full,
                src_transform=pre_src.transform,
                src_crs=pre_src.crs,
                dst_transform=post_src.transform,
                dst_crs=post_src.crs,
                resampling=rasterio.warp.Resampling.bilinear,
            )
        else:
            b08_pre_full = pre_src.read(4).astype(np.float32)
            b12_pre_full = pre_src.read(7).astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            nbr_pre  = np.where(
                b08_pre_full  + b12_pre_full  > 0,
                (b08_pre_full  - b12_pre_full)  / (b08_pre_full  + b12_pre_full  + 1e-6), 0.)
            nbr_post = np.where(
                b08_post + b12_post > 0,
                (b08_post - b12_post) / (b08_post + b12_post + 1e-6), 0.)

        dnbr      = (nbr_pre - nbr_post).astype(np.float32)
        burn_mask = (dnbr > DNBR_THRESHOLD).astype(np.uint8)

        burned_pct = burn_mask.mean() * 100
        print(f"  dNBR range   : [{dnbr.min():.3f}, {dnbr.max():.3f}]")
        print(f"  Burned area  : {burned_pct:.1f}% of tile")

        log_entries.append({
            "tile": tile,
            "pre_tif": pre_tif.name,
            "post_tif": post_tif.name,
            "dnbr_min": float(dnbr.min()),
            "dnbr_max": float(dnbr.max()),
            "burned_pct": float(burned_pct),
        })

        # Enumerate all valid patch positions
        rows = range(0, H - PATCH_SIZE + 1, PATCH_SIZE)
        cols = range(0, W - PATCH_SIZE + 1, PATCH_SIZE)

        fire_positions = []
        bg_positions   = []

        for r in rows:
            for c in cols:
                patch_mask  = burn_mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                patch_clear = mask_post[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

                cloud_frac = 1 - patch_clear.mean()
                if cloud_frac > MAX_CLOUD_FRAC:
                    total_skip_cloud += 1
                    continue

                fire_frac = patch_mask.mean()
                if fire_frac >= MIN_FIRE_FRAC:
                    fire_positions.append((r, c))
                else:
                    bg_positions.append((r, c))

        # Stratified sampling: balance fire vs background
        n_fire = len(fire_positions)
        n_bg   = int(n_fire * (1 - FIRE_RATIO) / FIRE_RATIO)
        n_bg   = min(n_bg, len(bg_positions))

        sampled_bg = (rng.choice(len(bg_positions), size=n_bg, replace=False).tolist()
                      if n_bg > 0 and bg_positions else [])
        selected = fire_positions + [bg_positions[i] for i in sampled_bg]

        print(f"  Fire patches : {n_fire}")
        print(f"  BG patches   : {n_bg} (from {len(bg_positions)} available)")
        print(f"  Total        : {len(selected)} patches")

        # Read all post-fire bands for patch extraction
        all_bands = post_src.read()   # (11, H, W)

        stem_prefix = f"{post_tif.stem[:25]}_{tile}"

        for r, c in tqdm(selected, desc=f"  Saving {tile}", leave=False):
            img_patch  = all_bands[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE].astype(np.int16)
            mask_patch = burn_mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE].astype(np.float32)

            name = f"{stem_prefix}_r{r:05d}_c{c:05d}.npy"
            np.save(PATCH_IMG_DIR  / name, img_patch)
            np.save(PATCH_MASK_DIR / name, mask_patch)
            total_written += 1

            is_fire = mask_patch.mean() >= MIN_FIRE_FRAC
            if is_fire:
                total_fire += 1
            else:
                total_bg += 1


# ── Phase C: save metadata ────────────────────────────────────────────────────
manifest = {
    "event": "Alexandroupolis wildfire, Evros, Greece — August 2023",
    "dnbr_threshold": DNBR_THRESHOLD,
    "patch_size": PATCH_SIZE,
    "fire_ratio_target": FIRE_RATIO,
    "total_patches": total_written,
    "fire_patches": total_fire,
    "background_patches": total_bg,
    "tiles": log_entries,
}
manifest_path = BASE / "data" / "greece" / "patches" / "manifest.json"
with open(manifest_path, "w") as fh:
    json.dump(manifest, fh, indent=2)

print()
print("=" * 65)
print("Phase B complete.")
print(f"  Patches written  : {total_written}")
print(f"    Fire           : {total_fire}  ({total_fire/max(total_written,1)*100:.1f}%)")
print(f"    Background     : {total_bg}")
print(f"  Skipped (cloud)  : {total_skip_cloud}")
print(f"  Manifest saved   : {manifest_path}")
print()
print("Drive Desktop will sync data/greece/ automatically.")
print("Next step:")
print("  Open notebooks/09_greece_zs_evaluation.ipynb in Google Colab (A100).")
print("  The Greece patches will be accessible at:")
print("  /content/drive/MyDrive/GeoAI/wildfire-spread/data/greece/patches/")
