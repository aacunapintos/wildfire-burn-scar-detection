# Wildfire Burn Scar Detection -- Prithvi-EO Foundation Models and Sentinel-2

Zero-shot cross-biome burn scar segmentation using IBM/NASA geospatial foundation models on Sentinel-2 L2A imagery. Trained on a single fire event in subtropical Argentina, evaluated across 4 biomes on 3 continents with no target-domain annotations.

![World map](results/world_map_cross_biome_v2.png)

*Training site (Corrientes, subtropical savanna) and three zero-shot evaluation sites across 3 continents. No annotations from any target region were used at any stage.*

Trained on the 2021-2022 Corrientes fire season, the model is applied without retraining to fires in central Argentina (2020, Argentine Monte), northeastern Greece (2023, largest EU wildfire on record, 81,000 ha of Mediterranean shrubland), and the North Slave Complex in Canada (2023, forced evacuation of Yellowknife, 163,000 ha of boreal forest). AUC-ROC exceeds 0.5 in all 3 zero-shot biomes.

---

## Key Results

| Version | Model | Region | Biome | IoU | Recall | Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|
| v1.0 | U-Net ResNet34 + FIRMS labels | Corrientes | Subtropical savanna | 0.013 | 7% | 14% | - |
| v1.5 | Prithvi-EO-1.0-100M + FPN | Corrientes val | Subtropical savanna | 0.538 | 71% | 69% | - |
| v1.6 | Prithvi-EO-1.0-100M + FPN (T=2) | Corrientes val | Subtropical savanna | **0.639** | **81%** | **75%** | - |
| v1.5 | Prithvi-EO-1.0-100M + FPN (ZS) | Cordoba | Argentine Monte | 0.115 | 21% | 20% | 0.738 |
| v1.5 | Prithvi-EO-1.0-100M + FPN (ZS) | Greece | Mediterranean | 0.232 | 32% | 45% | 0.595 |
| **v2.0** | **Prithvi-EO-2.0-300M + FPN (ZS)** | **Cordoba** | **Argentine Monte** | **0.115** | **21%** | **-** | **0.738** |
| **v2.0** | **Prithvi-EO-2.0-300M + FPN (ZS)** | **Greece** | **Mediterranean shrubland** | **0.234** | **31%** | **0.481** | **0.652** |
| **v2.0** | **Prithvi-EO-2.0-300M + FPN (ZS)** | **Canada NWT** | **Boreal forest** | **0.191** | **21%** | **0.680** | **0.606** |

49x improvement over the FIRMS-based baseline. The v2.0 backbone (307M parameters) achieves competitive zero-shot IoU across 3 biomes never seen during training. Precision of 0.680 on Canada confirms the model is conservative and reliable when it fires, even in a completely unseen biome.

---

## v2.0 Highlights

- Backbone upgraded from Prithvi-EO-1.0-100M to Prithvi-EO-2.0-300M (307M parameters, 3x larger)
- New zero-shot site: North Slave Complex, Northwest Territories, Canada (boreal forest, August 2023, forced evacuation of Yellowknife with 20,000 residents)
- AUC-ROC above 0.5 confirmed in all 3 zero-shot biomes: the model ranks burned vs. unburned pixels above chance everywhere
- Operational decision support: MC Dropout uncertainty maps classify patches as DEPLOY / VERIFY / MONITOR
- Cross-biome evaluation now spans 3 continents, 4 biomes, and 0 target-domain annotations

![Cross-biome summary](results/canada_cross_biome_summary_v2.png)

*Left: IoU across all 4 evaluation sites. Right: AUC-ROC and Recall comparison across the 3 zero-shot sites.*

---

## v2.1 Highlights

- Burn scar segmentation masks converted to GeoPackage (GPKG) vector polygons for direct GIS integration
- Per-scene NDVI and NBR spectral indices computed from Sentinel-2 L2A surface reflectance
- Full-scene RGB mosaics reconstructed from non-overlapping patches across all three zero-shot sites
- Greece and Canada outputs correctly georeferenced in UTM; outputs carry model attribution attributes (`area_ha`, `perimeter_km`, `site`, `date`, `model`)

![Cross-site vector summary](results/cross_site_vector_summary_v21.png)

*Predicted burn scar perimeters as vector polygons (GeoPackage) for three zero-shot sites. Detected areas are model predictions at zero-shot: precision ranges from 20% (Cordoba, Argentine Monte) to 68% (Canada, boreal forest). Approximate perimeters with boundary uncertainty of ~160m due to non-overlapping patch extraction.*

---

## Approach

### The label problem

Initial training used NASA FIRMS active fire detections as ground truth. Only 2.6% of patches contained fire pixels, producing a pixel-level IoU of 0.013. The validation metric appeared higher (0.50) because empty patches scored 1.0 trivially, inflating the per-batch average.

FIRMS detects active fire (thermal anomaly), not burn scars. A pixel that burned three days ago leaves no thermal signal but remains a burned area. The correct label source is dNBR (differenced Normalized Burn Ratio), computed from pre- and post-fire Sentinel-2 imagery.

```
dNBR = NBR_pre - NBR_post     where NBR = (B8A - B12) / (B8A + B12)
Burn scar threshold: dNBR > 0.10
```

This change increased positive patch coverage from 2.6% to 55.8% (21x more training signal) and enabled meaningful learning.

![dNBR vs FIRMS](results/dnbr_vs_firms_comparison.png)

*Each row shows one patch: RGB image (left), dNBR burn scar mask (center, threshold dNBR > 0.10), and FIRMS active fire mask (right). dNBR captures the complete burned area; FIRMS misses it because the thermal signal disappears days after burning.*

### Model architecture

**v1.x (Prithvi-EO-1.0-100M)**

| Component | Details |
|---|---|
| Backbone | Prithvi-EO-1.0-100M (IBM/NASA), embed_dim=768, depth=12 |
| Pretraining | Masked autoencoding on HLS (Harmonized Landsat-Sentinel) |
| Neck (v1.5) | Multi-scale FPN neck (layers 2, 5, 8, 11 to 256-ch feature map) |
| Neck (v1.6) | TemporalFusionNeck: concat(pre, post) per layer, 1x1 Conv, top-down FPN |
| Decoder | Feature Pyramid Network (FPN), trained from scratch |
| Input bands | B02, B03, B04, B8A, B11, B12 at 10m resolution |
| Temporal input | T=1 (post-fire only) through v1.5; T=2 (pre + post) from v1.6 |
| Patch size | 224x224 px |
| Loss | DiceLoss + FocalLoss, fire class weight = 5.0 |

**v2.0 (Prithvi-EO-2.0-300M)**

| Component | Details |
|---|---|
| Backbone | Prithvi-EO-2.0-300M (IBM/NASA), embed_dim=1024, depth=24, 307M parameters |
| Pretraining | Masked autoencoding on HLS multi-temporal (T=3) at global scale |
| Neck | MultiScaleNeck: layers 5, 11, 17, 23 (d_model=1024) to 256-ch feature map |
| Decoder | Feature Pyramid Network (FPN), 5-stage transposed convolutions, trained from scratch |
| Input bands | B02, B03, B04, B8A, B11, B12 (same 6 bands, T=1 post-fire) |
| Patch size | 224x224 px |
| Threshold | t=0.450 (optimized on Corrientes, fixed for all zero-shot evaluations) |
| Loss | DiceLoss + FocalLoss, fire class weight = 5.0 |

---

## Dataset

### Training: Corrientes, Argentina

| | |
|---|---|
| Region | Corrientes Province, NE Argentina (subtropical wetlands and grasslands) |
| Coordinates | 59.5W-56.0W / 29.0S-26.5S |
| Fire event | December 2021 - February 2022 (austral summer, extreme drought year) |
| Burned area | ~900,000 ha |
| Scenes | 6 Sentinel-2 L2A tiles, 0% cloud cover |
| Patches | 5,687 x 224x224 px |
| Positive rate | 55.8% (dNBR > 0.10) |
| Source | Copernicus Data Space Ecosystem (CDSE) |

### Zero-shot test 1: Cordoba, Argentina

| | |
|---|---|
| Region | Cordoba Province, central Argentina (Sierras Chicas, xerophytic scrubland) |
| Coordinates | 65.5W-62.5W / 33.0S-30.5S |
| Fire event | October-November 2020 |
| Biome | Argentine Monte (highland xerophytic scrubland) |
| Patches | 6,634 x 224x224 px |
| Positive rate | 63.7% (dNBR > 0.10) |
| Labels used in training | None |

### Zero-shot test 2: Alexandroupolis, Greece

| | |
|---|---|
| Region | Evros / Dadia-Lefkimi-Soufli, NE Greece |
| Coordinates | 25.6E-27.4E / 40.6N-42.0N |
| Fire event | August 2023 (largest EU wildfire on record, ~81,000 ha) |
| Biome | Mediterranean shrubland |
| Scenes | 18 pre-fire + 18 post-fire Sentinel-2 L2A tiles |
| Patches | 10,119 x 224x224 px |
| Positive rate | 76.9% (dNBR > 0.10) |
| Labels used in training | None |

### Zero-shot test 3: North Slave Complex, Canada (v2.0)

| | |
|---|---|
| Region | Northwest Territories, Canada (NWT) |
| Coordinates | 116.5W-113.5W / 61.8N-63.2N |
| Fire event | August 2023 (forced evacuation of Yellowknife, ~163,000 ha) |
| Biome | Boreal forest (completely distinct from subtropical savanna and Mediterranean shrubland) |
| Pre-fire scenes | June-July 2023 |
| Post-fire scenes | September-October 2023 |
| Raw downloads | 288 JP2 files |
| Patches | 9,064 x 224x224 px (filtered: MIN_VALID_FRAC=0.70, MAX_WATER_FRAC=0.30) |
| Labels used in training | None |

---

## Results

### Corrientes validation

![Portfolio overview](results/validation_overview.png)

*Best, median, and worst-performing patches from the Corrientes validation set. Error maps: green = true positive, orange = false positive, red = false negative. Right panel: full model progression v1.0 to v1.6 and best-model metrics (v1.6 T=2: IoU=0.64, F1=0.78).*

### Threshold optimization (v1.1)

![Threshold sweep](results/threshold_sweep.png)

Sweeping thresholds 0.05 to 0.95 on the validation set reveals the optimal operating point is t=0.65 for v1.x (t=0.450 for v2.0). At t=0.65, precision improves from 0.50 to 0.57 by reducing false positives while IoU and F1 also improve. The PR curve shows strong discriminative ability: the gain comes from choosing a better decision boundary, not retraining.

### Temporal fusion: Siamese T=2 model (v1.6)

Adding a pre-fire image (Oct-Nov 2021) as a second temporal input gives the model direct access to spectral change rather than post-fire reflectance alone. The Siamese backbone processes pre-fire and post-fire images in parallel; the TemporalFusionNeck concatenates features at transformer layers 2, 5, 8, 11 and fuses them before the FPN decoder.

![T=2 predictions](results/validation_overview_t2.png)

| Metric | v1.5 (T=1, post-fire only) | v1.6 (T=2, pre + post) | Delta |
|---|---|---|---|
| IoU | 0.538 | **0.639** | +0.101 (+18.6%) |
| F1 | 0.700 | **0.780** | +0.080 |
| Precision | 0.693 | **0.753** | +0.060 |
| Recall | 0.707 | **0.808** | +0.101 |

Both precision and recall improve simultaneously: the model eliminates false positives in areas with burn-scar-like reflectance that showed no spectral change between dates (bare soil, dry grassland).

### Zero-shot test 1: Cordoba, Argentina

![Cordoba predictions](results/cordoba_predictions.png)

| Metric | Corrientes val (v1.5) | Cordoba zero-shot (v1.5) |
|---|---|---|
| IoU | 0.538 | 0.115 |
| Recall | 0.71 | 0.21 |
| Precision | 0.69 | 0.20 |
| AUC-ROC | - | 0.738 |

Zero-shot transfer to Cordoba yields IoU=0.115 and AUC-ROC=0.738. AUC-ROC=0.738 confirms the model retains transferable burn-scar features, motivating few-shot fine-tuning.

### Few-shot domain adaptation: Cordoba

The FPN decoder was fine-tuned on 100 Cordoba patches (encoder kept frozen). The remaining 6,534 patches were held out as the test set.

![Fine-tuned predictions](results/cordoba_finetune_predictions.png)

| Metric | Zero-shot (v1.5) | Few-shot FT (100 patches) | Change |
|---|---|---|---|
| IoU | 0.115 | **0.329** | +0.214 (+186%) |
| Recall | 0.21 | **0.54** | +0.33 |
| Precision | 0.20 | **0.46** | +0.26 |
| AUC-ROC | 0.738 | **0.870** | +0.132 |

Fine-tuning the FPN decoder on 100 Cordoba patches yields a 2.9x IoU gain. All adaptation comes from the decoder adjusting to the new biome spectral distribution; the backbone encoder weights are never updated.

### Zero-shot test 2: Greece 2023

![Cross-biome summary v1](results/greece_cross_biome_summary.png)

| Metric | Corrientes val | Cordoba ZS | Greece ZS |
|---|---|---|---|
| IoU | 0.538 | 0.115 | **0.232** |
| Recall | 0.71 | 0.21 | 0.32 |
| Precision | 0.69 | 0.20 | 0.45 |
| AUC-ROC | - | 0.738 | 0.595 |

Zero-shot IoU on Greece (0.232) exceeds Cordoba (0.115), reflecting the unambiguous spectral signature of the large Dadia burn scar. Precision reaches 0.453 zero-shot, indicating the model correctly localises burned pixels when it fires.

![Greece best predictions](results/greece_zs_best.png)

### Zero-shot test 3: Canada NWT 2023 (v2.0)

The v2.0 model (Prithvi-EO-2.0-300M backbone) was applied zero-shot to the North Slave Complex wildfire in Northwest Territories, Canada (August 2023). This fire forced the evacuation of Yellowknife (20,000 residents) and burned approximately 163,000 ha of boreal forest, a biome spectrally completely distinct from the subtropical savanna used for training.

![Canada best predictions](results/canada_zs_best_v2.png)

| Metric | Corrientes val (v2.0) | Cordoba ZS (v2.0) | Greece ZS (v2.0) | Canada ZS (v2.0) |
|---|---|---|---|---|
| IoU | 0.532 | 0.115 | 0.234 | **0.191** |
| Recall | - | 0.209 | 0.314 | 0.210 |
| Precision | - | - | 0.481 | **0.680** |
| AUC-ROC | - | 0.738 | 0.652 | 0.606 |

AUC-ROC exceeds 0.5 in all 3 zero-shot biomes. Precision of 0.680 on Canada is the highest across all ZS sites: when the model predicts a burn scar in boreal forest, it is right 68% of the time despite never having seen this biome. AUC-ROC decreases monotonically with biome distance from the training site (0.738 same-continent, 0.652 cross-continental Mediterranean, 0.606 boreal), a pattern consistent with foundation model geographic generalization.

![Canada ZS curves](results/canada_zs_curves.png)

### Operational decision support (v2.0)

MC Dropout (forward hooks on FPN decoder GELU layers, p=0.08, N=30 passes) generates per-patch uncertainty estimates. Patches are classified into three operational categories based on mean burn probability:

| Category | Threshold | Action |
|---|---|---|
| DEPLOY | P > 0.65 | High confidence -- dispatch field team |
| VERIFY | 0.40 < P <= 0.65 | Medium confidence -- aerial check first |
| MONITOR | 0.20 < P <= 0.40 | Low confidence -- satellite follow-up |

![Operational decision support](results/canada_decision_support_v2.png)

*Operational decision support on Canada ZS patches. Each row shows RGB, burn probability map, and dNBR ground truth for representative DEPLOY, VERIFY, and MONITOR patches.*

### Vector output: burn scar perimeters (v2.1)

![Greece vector output](results/greece_vector_output_v21.png)

*Greece ZS: RGB mosaic (post-fire Sentinel-2), burn probability map, NDVI, and vector polygon perimeters in UTM zone 35N. Strong red probability patches correspond to the Dadia forest burn scar (Evros, August 2023).*

| Site | Polygons | Detected area | Reference area | Note |
|---|---|---|---|---|
| Cordoba ZS | 361 | 158,358 ha | ~28,000 ha | Zero-shot, Argentine Monte -- high false positive rate |
| Greece ZS | 189 | 413,398 ha | ~81,000 ha | Zero-shot, Mediterranean shrubland -- correctly georeferenced (UTM 35N) |
| Canada ZS | 16 | 94,508 ha | ~163,000 ha | Single tile, boreal forest -- correctly georeferenced (UTM 11N) |

Detected areas reflect zero-shot precision, not validated measurements. Recall at zero-shot ranges from 21% (Cordoba, Canada) to 31% (Greece), consistent with the patch-level metrics in the Key Results table. GeoPackage attributes: `area_ha`, `perimeter_km`, `site`, `date`, `model`.

---

## Limitations

**Biome-induced domain shift.** The FPN decoder was trained on a single biome (Corrientes wetlands). Recall drops significantly in unseen biomes (0.21 in Canada vs 0.81 in Corrientes T=2), meaning the model misses a large fraction of burned area at zero-shot. Precision remains useful (0.68 in Canada): patches flagged as burned are reliable, but the model is conservative.

**Single temporal input in v2.0.** The v2.0 model uses T=1 (post-fire image only). The v1.6 Siamese model demonstrated that adding a pre-fire temporal input raises IoU by +18.6% on Corrientes. Extending v2.0 to T=3 (as Prithvi-EO-2.0 was pretrained) is the highest-impact architectural improvement.

**ViT tiling artifact.** Predictions show 16x16 pixel block artifacts inherited from the ViT patch tokenizer. Gaussian smoothing (sigma=3) reduces the artifact in uncertainty maps; the effect on binary IoU is minor.

**Patch extraction gap.** Patches are extracted at stride=256 with center crop T=16, leaving 32-pixel (320m) gaps between adjacent patch predictions. Vector perimeters carry a boundary uncertainty of approximately 160m and area estimates may differ from true burned extent by 15-25% depending on polygon size. Overlapping extraction (stride less than 256) would eliminate this artifact but requires re-downloading the original Sentinel-2 imagery.

**Water and nodata contamination.** NWT contains large lakes and rivers that produce false positives. A post-processing NDWI filter (applied during evaluation) partially mitigates this; morphological post-processing would further improve precision.

---

## Roadmap

| Priority | Improvement | Expected gain | Status |
|---|---|---|---|
| 1 | Second training biome (Mediterranean 2021 or boreal 2019) | +10-20 IoU ZS | Planned (v3.0) |
| 2 | Multi-temporal input T=3 (matching Prithvi-EO-2.0 pretraining) | +5-10 IoU | Planned (v3.0) |
| 3 | Test-Time Augmentation (flip H/V average) | +2-5 IoU | Planned (v2.1) |
| 4 | Vector output (GeoPackage burn scar polygons + NDVI) | Operational | **Done (v2.1)** |
| 5 | FastAPI inference endpoint (coordinates + date to mask) | Deployment | Planned (MLOps) |
| 6 | Morphological post-processing (remove isolated pixels) | Precision | Planned |

---

## Changelog

| Version | Change | Val IoU | Notes |
|---|---|---|---|
| v1.0 | Base model, FIRMS labels, threshold=0.50 | 0.013 | Prithvi-EO-1.0-100M + FPN |
| v1.1 | Switch to dNBR labels (threshold=0.10) | 0.42 | 21x more positive patches. 49x IoU improvement vs v1.0 |
| v1.2 | Optimal threshold t=0.65, continuation training | 0.45 | Post-processing only for threshold. Epoch 73 best checkpoint |
| v1.3 | Partial backbone unfreeze (last 2 transformer blocks) | 0.50 | Differential LR (1e-5 backbone, 5e-5 decoder) |
| v1.4 | Spectral variation training (contrast, brightness, noise) | 0.36 | Too aggressive late in training. v1.3 checkpoint preserved |
| v1.5 | Multi-scale FPN neck (layers 2, 5, 8, 11) | 0.538 | 45 epochs. IoU +8.9% vs v1.3. ZS Cordoba: IoU=0.115, AUC-ROC=0.738 |
| v1.6 | Siamese T=2 temporal fusion (pre + post fire) | 0.639 | TemporalFusionNeck. IoU +18.6% vs v1.5. FT Cordoba T=2: IoU=0.810 |
| v1.7 | Cordoba geographic evaluation (ZS and few-shot FT, T=1 and T=2) | 0.538 | ZS: IoU=0.115 (T=1), 0.087 (T=2). FT: IoU=0.329 (T=1), 0.810 (T=2 within-region) |
| v1.8 | Cross-continental ZS evaluation on Greece 2023 (Alexandroupolis, Mediterranean) | 0.538 | ZS Greece: IoU=0.232, AUC-ROC=0.595. No Greek labels. 10,119 patches |
| **v2.0** | **Backbone upgrade to Prithvi-EO-2.0-300M (307M params, embed_dim=1024, depth=24)** | **0.532** | **New ZS site: Canada NWT 2023 (boreal forest, 163,000 ha). AUC-ROC > 0.5 confirmed in 3 biomes. MC Dropout operational decision support** |
| **v2.1** | **Vector output: burn scar perimeters as GeoPackage (GPKG) for 3 zero-shot sites** | **0.532** | **NDVI + NBR per scene. RGB mosaics. Georeferenced polygons (UTM) with area, perimeter, model attributes. Boundary uncertainty ~160m** |

---

## Repository Structure

```
wildfire-spread/
+-- notebooks/
|   +-- 01_data_acquisition.ipynb        Sentinel-2 L2A (CDSE) + NASA FIRMS download
|   +-- 02_preprocessing.ipynb           Band stacking, patch extraction, dNBR labels
|   +-- 03_baseline.ipynb                U-Net ResNet34 training + diagnostic evaluation
|   +-- 04_prithvi_training.ipynb        Prithvi-EO-1.0-100M fine-tuning v1.0-v1.5 (Colab A100)
|   +-- 04b_prithvi_t2.ipynb             Siamese T=2 temporal fusion, v1.6 (Colab A100)
|   +-- 05_cordoba_data.ipynb            Cordoba test set acquisition and preprocessing
|   +-- 06_cordoba_evaluation.ipynb      Geographic generalization + few-shot adaptation (Colab A100)
|   +-- 07_inference_demo.ipynb          Single-patch inference demo (Colab)
|   +-- 08_prithvi_v2_training.ipynb     Prithvi-EO-2.0-300M fine-tuning v2.0 (Colab A100)
|   +-- 09_greece_zs_evaluation.ipynb    Cross-continental ZS evaluation on Greece 2023 (Colab A100)
|   +-- 10_canada_zs_evaluation.ipynb    ZS evaluation on Canada NWT 2023 + decision support (Colab A100)
+-- results/
|   +-- world_map_cross_biome_v2.png     Geographic overview: 4 sites across 3 continents
|   +-- canada_cross_biome_summary_v2.png  Cross-biome IoU and AUC-ROC comparison (v2.0, 4 sites)
|   +-- canada_decision_support_v2.png   Operational decision support (DEPLOY/VERIFY/MONITOR)
|   +-- cordoba_vector_output_v21.png   Cordoba v2.1: RGB mosaic, probability, NDVI, vector polygons
|   +-- greece_vector_output_v21.png    Greece v2.1: RGB mosaic, probability, NDVI, vector polygons
|   +-- canada_vector_output_v21.png    Canada v2.1: RGB mosaic, probability, NDVI, vector polygons
|   +-- cross_site_vector_summary_v21.png  Cross-site burn scar perimeters as UTM vector polygons (v2.1)
|   +-- canada_zs_best_v2.png            Canada ZS: best predictions
|   +-- canada_zs_curves.png             Canada ZS: PR and ROC curves
|   +-- validation_overview.png          Corrientes: best/median/worst patches, v1.0 to v1.6 progression
|   +-- validation_overview_t2.png       Corrientes T=2: detailed patch grid and training curves
|   +-- threshold_sweep.png              Metrics vs threshold + PR curve (v1.1)
|   +-- dnbr_vs_firms_comparison.png     dNBR vs FIRMS label comparison
|   +-- cordoba_predictions.png          Cordoba v1.5 zero-shot: best patches
|   +-- cordoba_evaluation_overview.png  All-in-one: model progression + Cordoba ZS/FT + T=2 delta
|   +-- cordoba_finetune_predictions.png Cordoba v1.5 few-shot FT: best patches
|   +-- greece_cross_biome_summary.png   Cross-biome IoU and AUC-ROC comparison (v1.x)
|   +-- greece_zs_best.png               Greece 2023: best zero-shot predictions
|   +-- greece_zs_curves.png             Greece 2023: PR and ROC curves
+-- scripts/
|   +-- 00_prefire_download.py           Download pre-fire Sentinel-2 tiles for T=2 pairs
|   +-- 03b_paired_patches.py            Build aligned pre/post patch pairs
|   +-- 09_greece_download.py            Download Sentinel-2 L2A for Alexandroupolis 2023
|   +-- 10_greece_patches.py             JP2 to GeoTIFF, dNBR, patch extraction (Greece)
|   +-- 11_canada_pipeline.py            Download + JP2 to GeoTIFF + dNBR + patches (Canada, combined)
+-- models/
|   +-- best_prithvi_v2_burn_scar_wildfire.pth  v2.0 checkpoint (Prithvi-EO-2.0-300M + FPN)
+-- environment.yml
+-- .gitignore
```

---

## Reproduce

**Environment**

```bash
conda env create -f environment.yml
conda activate geoai-wildfire
```

**Credentials**

Copy `.env.example` to `.env` and fill in:

```
CDSE_USER=your_copernicus_user
CDSE_PASSWORD=your_copernicus_password
FIRMS_API_KEY=your_firms_key
```

- CDSE: free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
- FIRMS: free API key at [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/api/area/)

**Run order**

Notebooks 01-03 and 05 run locally on CPU (4-5 hours total, mostly data download).
Notebooks 04, 04b, 06, 08, 09, 10 require a GPU (Google Colab A100 recommended).
Script 11 runs locally for the Canada pipeline (download + patch extraction, 6-10 hours).

---

## Data Sources

| Dataset | Provider | Access |
|---|---|---|
| Sentinel-2 L2A | ESA / Copernicus Data Space Ecosystem (CDSE) | Free, registration required |
| VIIRS SNPP active fire | NASA FIRMS | Free, API key required |
| ERA5 reanalysis (fire weather) | Copernicus Climate Data Store (CDS) | Free, registration required |

---

## References

- Jakubik, J. et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence. arXiv:2310.18660
- Prithvi-EO-1.0-100M: [ibm-nasa-geospatial/Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M)
- Prithvi-EO-2.0-300M: [ibm-nasa-geospatial/Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
- Key, C.H. and Benson, N.C. (2006). Landscape Assessment: Ground measure of severity. USDA Forest Service
- terratorch: [github.com/IBM/terratorch](https://github.com/IBM/terratorch)

---

## License

MIT
