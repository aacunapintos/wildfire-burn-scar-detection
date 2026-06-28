"""
Pre-fire preprocessing + spatially aligned paired patch extraction.

Reads:
  data/sentinel2/raw_prefire/   (downloaded by 00_prefire_download.py)
  data/sentinel2/raw/           (existing post-fire tiles)
  data/patches/images/          (existing post-fire patches)
  data/patches/masks_dnbr/      (existing dNBR labels)

Writes:
  data/sentinel2/processed_prefire/<stem>.tif   (reprojected pre-fire GeoTIFFs)
  data/patches/images_prefire/<name>.npy         (same filename as post-fire patch)

Naming convention for output pairs:
  images/          20240507T114228_20220128_r00256_c01024.npy   (post-fire, Jan 2022)
  images_prefire/  20240507T114228_20220128_r00256_c01024.npy   (pre-fire, Oct/Nov 2021, SAME location)
  masks_dnbr/      20240507T114228_20220128_r00256_c01024.npy   (dNBR label)

Run from project root:
    python scripts/03b_paired_patches.py

Requirements: rasterio, numpy, tqdm (same conda env as notebook 02)
Estimated time: 30-60 min on local CPU.
"""

import os
import sys
import re
import json
import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows
import rasterio.transform
from rasterio.crs import CRS
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ── PROJ fix (same as notebook 02) ───────────────────────────────────────────
_conda_prefix = Path(sys.executable).parent.parent
_proj_data = _conda_prefix / "Library" / "share" / "proj"
if _proj_data.exists():
    os.environ["PROJ_DATA"] = str(_proj_data)
    os.environ["PROJ_LIB"]  = str(_proj_data)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
RAW_PRE_DIR  = BASE / "data" / "sentinel2" / "raw_prefire"
RAW_POST_DIR = BASE / "data" / "sentinel2" / "raw"
PROC_PRE_DIR = BASE / "data" / "sentinel2" / "processed_prefire"
POST_IMG_DIR = BASE / "data" / "patches" / "images"
PRE_IMG_DIR  = BASE / "data" / "patches" / "images_prefire"
MASK_DIR     = BASE / "data" / "patches" / "masks_dnbr"

PROC_PRE_DIR.mkdir(parents=True, exist_ok=True)
PRE_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (same as notebook 02) ──────────────────────────────────────────
CRS_UTM        = CRS.from_epsg(32721)
BANDS          = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
PATCH_SIZE     = 256
MAX_CLOUD_FRAC = 0.3   # more permissive for pre-fire scenes
SCL_CLEAR      = {4, 5, 6}

# ── Tile ID extraction ────────────────────────────────────────────────────────
def tile_id_from_name(name):
    m = re.search(r"_T(\d{2}[A-Z]{3})_", name)
    return f"T{m.group(1)}" if m else None


# ══════════════════════════════════════════════════════════════════════════════
# Phase A — Process raw pre-fire JP2 tiles to reprojected GeoTIFFs
# ══════════════════════════════════════════════════════════════════════════════

def reproject_band(src_path, dst_crs, resolution=10):
    with rasterio.open(src_path) as src:
        src_crs = src.crs if src.crs else CRS.from_epsg(32721)
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
        return np.where(denom != 0, (a - b) / denom, 0.0).astype(np.float32)


def process_prefire_scene(scene_dir, out_dir):
    """
    Process one pre-fire scene directory.
    scene_dir: Path to <item_id>/<date>/
    Output: <stem>.tif with same 11-band format as notebook 02.
    """
    item_id  = scene_dir.parent.name
    date_str = scene_dir.name
    stem     = f"{item_id[-15:]}_{date_str}"
    out_path = out_dir / f"{stem}.tif"

    if out_path.exists():
        print(f"  SKIP {out_path.name} (already processed)")
        return out_path

    # SCL mask
    scl_path = scene_dir / "SCL.jp2"
    if not scl_path.exists():
        print(f"  SKIP {stem} — SCL.jp2 missing")
        return None
    scl_data, scl_transform = reproject_band(scl_path, CRS_UTM, resolution=10)
    clear_mask = np.isin(scl_data, list(SCL_CLEAR))
    shape = scl_data.shape

    # Spectral bands
    band_data = {}
    for band in BANDS:
        jp2 = scene_dir / f"{band}.jp2"
        if not jp2.exists():
            print(f"  WARN {stem} — {band}.jp2 missing, filling with zeros")
            band_data[band] = np.zeros(shape, dtype=np.int16)
            continue
        data, _ = reproject_band(jp2, CRS_UTM, resolution=10)
        if data.shape != shape:
            # Crop or pad to match SCL shape
            min_h = min(data.shape[0], shape[0])
            min_w = min(data.shape[1], shape[1])
            aligned = np.zeros(shape, dtype=data.dtype)
            aligned[:min_h, :min_w] = data[:min_h, :min_w]
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
            date=date_str, item_id=item_id,
        )

    pct_clear = clear_mask.mean() * 100
    size_mb   = out_path.stat().st_size / 1e6
    print(f"  OK  {out_path.name}  ({size_mb:.0f} MB, {pct_clear:.1f}% clear)")
    return out_path


print("=" * 62)
print("Phase A — Processing pre-fire raw tiles to GeoTIFFs")
print("=" * 62)

prefire_scenes = []
for item_dir in sorted(RAW_PRE_DIR.iterdir()):
    if not item_dir.is_dir():
        continue
    for date_dir in sorted(item_dir.iterdir()):
        if date_dir.is_dir() and list(date_dir.glob("*.jp2")):
            prefire_scenes.append(date_dir)

if not prefire_scenes:
    sys.exit(
        "ERROR: No pre-fire JP2 tiles found in:\n"
        f"  {RAW_PRE_DIR}\n\n"
        "Run 00_prefire_download.py first."
    )

print(f"Found {len(prefire_scenes)} pre-fire scene(s):")
prefire_tifs = {}
for scene_dir in prefire_scenes:
    item_id  = scene_dir.parent.name
    tile_id  = tile_id_from_name(item_id)
    print(f"  {tile_id}  {scene_dir.name}  ({item_id[-20:]})")
    tif = process_prefire_scene(scene_dir, PROC_PRE_DIR)
    if tif and tile_id:
        prefire_tifs.setdefault(tile_id, []).append(tif)

print(f"\nPre-fire GeoTIFFs ready: {sum(len(v) for v in prefire_tifs.values())}")
for tile, paths in prefire_tifs.items():
    for p in paths:
        print(f"  {tile}  {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase B — Extract spatially aligned pre-fire patches
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 62)
print("Phase B — Extracting aligned pre-fire patches")
print("=" * 62)

# Build mapping: post-fire tile_id → post-fire GeoTIFF (from existing raw dir)
post_tif_dir = BASE / "data" / "sentinel2" / "processed"
post_tifs_by_tile = defaultdict(list)
for tif in sorted(post_tif_dir.glob("*.tif")):
    if "firemask" in tif.name:
        continue
    m = re.search(r"_T(\d{2}[A-Z]{3})_", tif.stem)
    if m:
        tile_id = f"T{m.group(1)}"
    else:
        # Try matching from stem pattern: <proc_date>_<acq_date>.tif
        # Fall back: just store it
        tile_id = tif.stem[:15]
    post_tifs_by_tile[tile_id].append(tif)

# Also build post-patch → tile mapping from patch filenames
# Patch name = {proc_date}_{acq_date}_r{R}_c{C}.npy
# proc_date is item_id[-15:], we can match to tiles via post_tif

# Build: stem → tile_id for all post-fire patches
post_patches = sorted(POST_IMG_DIR.glob("*.npy"))
if not post_patches:
    sys.exit(f"ERROR: No post-fire patches found in {POST_IMG_DIR}")

print(f"Post-fire patches: {len(post_patches)}")

# For each post-fire patch, find the corresponding post-fire tif and pre-fire tif
# Strategy: match stem prefix to tile_id via post-fire GeoTIFF filenames

# Build stem → post-tif path
stem_to_posttif = {}
for tif in post_tif_dir.glob("*.tif"):
    if "firemask" in tif.name:
        continue
    stem_to_posttif[tif.stem] = tif

# Build tile_id → post-tif stem mapping
# post-tif stem = "{item_id[-15:]}_{date}"  where item_id contains tile_id
posttif_to_tile = {}
for tif in post_tif_dir.glob("*.tif"):
    if "firemask" in tif.name:
        continue
    # Try to recover tile from raw dir item names
    for item_dir in sorted(RAW_POST_DIR.iterdir()):
        if not item_dir.is_dir():
            continue
        tile_id = tile_id_from_name(item_dir.name)
        if tile_id is None:
            continue
        # Match: item_id[-15:] appears in tif stem
        if item_dir.name[-15:] in tif.stem:
            posttif_to_tile[tif.stem] = tile_id
            break

print(f"Post-tif → tile mapping: {len(posttif_to_tile)} entries")
for stem, tile in sorted(posttif_to_tile.items()):
    print(f"  {tile}  {stem}")


def extract_window_from_tif(src, bounds_xy, patch_size):
    """
    Extract a patch from `src` (open rasterio file) at the geographic
    bounds given by bounds_xy = (left, bottom, right, top) in src.crs.
    Returns numpy array (C, patch_size, patch_size) or None if out of bounds.
    """
    win = rasterio.windows.from_bounds(*bounds_xy, transform=src.transform)
    col_off = int(round(win.col_off))
    row_off = int(round(win.row_off))
    width   = int(round(win.width))
    height  = int(round(win.height))

    # Clamp to image bounds
    if col_off < 0 or row_off < 0:
        return None
    if col_off + patch_size > src.width or row_off + patch_size > src.height:
        return None

    window = rasterio.windows.Window(col_off, row_off, patch_size, patch_size)
    try:
        data = src.read(window=window)
    except Exception:
        return None

    if data.shape[1] != patch_size or data.shape[2] != patch_size:
        return None
    return data


def get_patch_bounds(post_tif_path, row, col, patch_size):
    """Get geographic bounds (left,bottom,right,top) of a patch in a GeoTIFF."""
    with rasterio.open(post_tif_path) as src:
        win = rasterio.windows.Window(col, row, patch_size, patch_size)
        bounds = rasterio.windows.bounds(win, src.transform)
    return bounds  # (left, bottom, right, top)


def parse_row_col(patch_name):
    """Extract row and col offsets from patch filename."""
    m = re.search(r"_r(\d+)_c(\d+)", patch_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def get_patch_stem_prefix(patch_name):
    """Return the part before _rXXXXX e.g. '20221228T160256_20211231'."""
    m = re.search(r"^(.+?)_r\d+_c\d+\.npy$", patch_name)
    return m.group(1) if m else None


# Group patches by post-tif stem for efficiency
patches_by_stem = defaultdict(list)
for p in post_patches:
    stem_prefix = get_patch_stem_prefix(p.name)
    if stem_prefix:
        patches_by_stem[stem_prefix].append(p)

total_written = 0
total_skipped_cloud = 0
total_skipped_bounds = 0
total_skipped_no_prefire = 0

for stem_prefix, patch_list in sorted(patches_by_stem.items()):
    tile_id = posttif_to_tile.get(stem_prefix)
    if tile_id is None:
        print(f"  WARN: no tile_id for stem '{stem_prefix}' — skipping {len(patch_list)} patches")
        total_skipped_no_prefire += len(patch_list)
        continue

    if tile_id not in prefire_tifs or not prefire_tifs[tile_id]:
        print(f"  WARN: no pre-fire GeoTIFF for {tile_id} — skipping {len(patch_list)} patches")
        total_skipped_no_prefire += len(patch_list)
        continue

    # Use first (clearest) pre-fire scene for this tile
    pre_tif_path  = prefire_tifs[tile_id][0]
    post_tif_path = BASE / "data" / "sentinel2" / "processed" / f"{stem_prefix}.tif"

    if not post_tif_path.exists():
        print(f"  WARN: post-fire GeoTIFF not found: {post_tif_path.name}")
        total_skipped_no_prefire += len(patch_list)
        continue

    print(f"\n{tile_id}  {stem_prefix}  ({len(patch_list)} patches)")
    print(f"  Pre-fire tif : {pre_tif_path.name}")

    with rasterio.open(pre_tif_path) as pre_src:
        for patch_path in tqdm(patch_list, desc=f"  {tile_id}", leave=False):
            out_path = PRE_IMG_DIR / patch_path.name
            if out_path.exists():
                continue

            row, col = parse_row_col(patch_path.name)
            if row is None:
                continue

            # Get geographic bounds of this post-fire patch
            bounds = get_patch_bounds(post_tif_path, row, col, PATCH_SIZE)
            if bounds is None:
                total_skipped_bounds += 1
                continue

            # Extract the same geographic area from pre-fire tif
            pre_patch = extract_window_from_tif(pre_src, bounds, PATCH_SIZE)
            if pre_patch is None:
                total_skipped_bounds += 1
                continue

            # Cloud check on pre-fire patch (band index 10 = MASK)
            if pre_patch.shape[0] > 10:
                cloud_frac = 1 - pre_patch[10].mean()
                if cloud_frac > MAX_CLOUD_FRAC:
                    total_skipped_cloud += 1
                    continue

            np.save(out_path, pre_patch.astype(np.int16))
            total_written += 1

print()
print("=" * 62)
print(f"Phase B complete.")
print(f"  Pre-fire patches written  : {total_written}")
print(f"  Skipped (cloud/invalid)   : {total_skipped_cloud}")
print(f"  Skipped (out of bounds)   : {total_skipped_bounds}")
print(f"  Skipped (no pre-fire tif) : {total_skipped_no_prefire}")

# Compute valid pairs: must have both post-fire and pre-fire patch + dNBR mask
paired = []
for f in sorted(PRE_IMG_DIR.glob("*.npy")):
    post_f = POST_IMG_DIR / f.name
    mask_f = MASK_DIR      / f.name
    if post_f.exists() and mask_f.exists():
        paired.append(f.name)

print(f"  Valid T=2 pairs (pre+post+mask): {len(paired)}")

# Save pair list for training notebook
pair_manifest = BASE / "data" / "patches" / "t2_pairs.json"
with open(pair_manifest, "w") as fh:
    json.dump(paired, fh, indent=2)
print(f"  Pair manifest saved: {pair_manifest}")

print()
print("Next step:")
print("  Sync data/patches/images_prefire/ to Google Drive, then")
print("  open notebooks/04b_prithvi_t2.ipynb in Colab (A100).")
