# Wildfire Burn Scar Detection -- Prithvi-EO Foundation Models and Sentinel-2

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://aacunapintos.github.io/wildfire-burn-scar-detection/) [![Version](https://img.shields.io/badge/version-v2-orange)](CHANGELOG.md) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](#license)

<p>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/TorchGeo-4CAF50?style=flat-square&logoColor=white" alt="TorchGeo"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
<img src="https://img.shields.io/badge/GDAL-5C8A3C?style=flat-square&logoColor=white" alt="GDAL"/>
<img src="https://img.shields.io/badge/Rasterio-8B7355?style=flat-square&logoColor=white" alt="Rasterio"/>
<img src="https://img.shields.io/badge/GeoPandas-139467?style=flat-square&logoColor=white" alt="GeoPandas"/>
<img src="https://img.shields.io/badge/Shapely-6B4FA0?style=flat-square&logoColor=white" alt="Shapely"/>
<img src="https://img.shields.io/badge/segmentation--models--pytorch-662D91?style=flat-square&logoColor=white" alt="segmentation-models-pytorch"/>
<img src="https://img.shields.io/badge/Prithvi--EO--2.0-054ADA?style=flat-square&logo=ibm&logoColor=white" alt="Prithvi-EO-2.0"/>
<img src="https://img.shields.io/badge/terratorch-0F62FE?style=flat-square&logo=ibm&logoColor=white" alt="terratorch"/>
<img src="https://img.shields.io/badge/Sentinel--2%20%2F%20Copernicus-00205B?style=flat-square&logoColor=white" alt="Sentinel-2 / Copernicus"/>
<img src="https://img.shields.io/badge/NASA%20FIRMS-0B3D91?style=flat-square&logo=nasa&logoColor=white" alt="NASA FIRMS"/>
<img src="https://img.shields.io/badge/Leaflet-199900?style=flat-square&logo=leaflet&logoColor=white" alt="Leaflet"/>
<img src="https://img.shields.io/badge/GitHub%20Pages-222222?style=flat-square&logo=githubpages&logoColor=white" alt="GitHub Pages"/>
<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter"/>
</p>

Zero-shot cross-biome burn scar segmentation using IBM/NASA geospatial foundation models on Sentinel-2 L2A imagery. Trained on two fire events on two continents, the model generates GIS-ready burn scar polygons on wildfires it has never seen, with no target-domain annotations.

**[Explore the live interactive demo](https://aacunapintos.github.io/wildfire-burn-scar-detection/)** -- click any detected polygon on the Valparaiso, Chile 2023 wildfire for its burn probability, confidence level, area, and perimeter.

![World map](results/world_map_v22.png)

*Training sites (Corrientes, Argentina, and East Gippsland, Australia) and all four zero-shot evaluation sites to date, across 3 continents: Cordoba (Argentine Monte), Greece (Mediterranean shrubland), Canada (boreal forest), and Chile (Mediterranean wildland-urban interface, current showcase). Full per-site figures and metrics: [CHANGELOG.md](CHANGELOG.md).*

---

## Why this project

I wanted to know whether a large pretrained Earth observation foundation model could be pointed at a wildfire it had never seen, in a biome it had never trained on, and still produce something a disaster-response team could act on without months of region-specific labeling. NASA FIRMS active-fire data turned out to be the wrong signal for that: it detects heat, not burn scars, and misses most of the burned area once the thermal anomaly fades. That led to dNBR-based labels instead, and then to testing the resulting model zero-shot across four increasingly different biomes on three continents, to find out concretely where that promise holds and where it breaks.

## How it works

Detects wildfire burn scars from Sentinel-2 imagery and exports GIS-ready polygons (GeoPackage) for downstream use: post-fire damage assessment, insurance claims validation, land management planning, and government disaster-response reporting.

| | |
|---|---|
| Backbone | Prithvi-EO-2.0-300M (IBM/NASA), 307M parameters, pretrained on global multi-temporal HLS imagery |
| Neck + decoder | Multi-scale FPN neck (transformer layers 5/11/17/23) into a 5-stage transposed-convolution decoder, trained from scratch |
| Input | Sentinel-2 L2A, 6 bands (B02, B03, B04, B8A, B11, B12), 224x224 px patches |
| Labels | dNBR (differenced Normalized Burn Ratio) from pre/post-fire imagery, not raw active-fire detections |
| Output | Per-pixel burn probability, vectorized into GeoPackage polygons with area, perimeter, and a confidence tier |

Full architecture table and the FIRMS-to-dNBR pivot story: [CHANGELOG.md](CHANGELOG.md).

## Latest version (v2.3)

Re-evaluated the three earlier zero-shot sites (Cordoba, Greece, Canada) against the current checkpoint for the first time since a second training biome (Australia) was added, and introduced an unsupervised per-scene adaptive threshold (Otsu) as the recommended way to operate the model on a new region. The result is not the uniform improvement the Roadmap used to assume: Greece improved, Cordoba regressed, and one apparent gain in Canada turned out to be a measurement artifact, caught before it reached this page. Full breakdown in Limitations below and in [CHANGELOG.md](CHANGELOG.md).

More training data alone does not reliably fix cross-region generalization. I am spending part of this summer on a v3 redesign that targets the actual root causes instead (see Roadmap): burn severity labels that do not transfer across vegetation types, and a model input that depends on absolute reflectance rather than relative change. Targeting results by the end of summer 2026.

---

## Key Results

**49x improvement over the FIRMS-based baseline**: pixel-level IoU went from 0.013 (naive active-fire labels) to 0.6512 (dNBR labels + Prithvi-EO-2.0-300M, threshold-tuned) on held-out validation data.

| Site | Role | IoU (fixed t=0.45) | AUC-ROC | Note |
|---|---|---|---|---|
| Corrientes + Australia | Training | 0.651 | -- | Held-out validation split |
| Cordoba, Argentina | Zero-shot | 0.105 | 0.666 | Regressed after adding Australia (see Limitations) |
| Greece | Zero-shot | 0.245 (0.257 w/ Otsu) | 0.668 | Improved after adding Australia |
| Canada | Zero-shot | 0.289 | 0.646 | See Limitations for a caught measurement artifact |
| Chile (current showcase) | Zero-shot | 0.175 | 0.855 | 146 polygons, 203,910 ha -- see Results below |

Full per-version metrics, tables, and figures: **[CHANGELOG.md](CHANGELOG.md)**. "v2" is the public release name for this arc of work; internally it was built across incremental engineering iterations (v2.0-v2.3), each tagged and logged separately for anyone who wants the granular history.

---

## Results

### Vector output: Chile 2023 (v2)

![Chile vector output](results/chile_vector_output_v22.png)

*Chile ZS: RGB mosaic (post-fire Sentinel-2), burn probability map, and vector polygon perimeters in UTM zone 19S, Valparaiso Region.*

| Site | Polygons | Detected area | Mean burn probability | IoU | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Chile ZS | 146 | 203,910 ha | 0.44 (range 0.31-0.69) | 0.175 | 0.855 | 0.218 | 0.472 |

Metrics computed pixel-wise against a dNBR (>0.15) ground truth raster for the most-burned tile (T19HBD), at the same decision threshold (P>0.450) used for the vector output -- 116,976,640 valid pixels. AUC-ROC of 0.855 is the highest of the four zero-shot sites tested to date: strong ranking discrimination, even though the fixed threshold is not precision-optimal for this biome (see Limitations).

GeoPackage attributes: `area_ha`, `perimeter_km`, `site`, `date`, `model`, `mean_prob`, `mean_dnbr`, `landcover_name`, `landcover_pct`. Explore results interactively on the [live dashboard](https://aacunapintos.github.io/wildfire-burn-scar-detection/): each polygon shows its burn probability, confidence tier (HIGH / MEDIUM / LOW), land cover, area, and perimeter on click. dNBR is in the GeoPackage but not yet on the dashboard; held back pending the v3 redesign (see Roadmap and Limitations).

Detailed per-site results for Cordoba, Greece, and Canada (vector output, PR/ROC curves, operational decision support): [CHANGELOG.md](CHANGELOG.md#detailed-results).

---

## Limitations

**Biome-induced domain shift.** Recall drops significantly in unseen biomes (0.21-0.34 zero-shot vs. 0.81 in-domain with T=2), meaning the model misses a large fraction of burned area outside its training biomes. Precision is generally more reliable: patches flagged as burned tend to be correct, but the model is conservative.

**Single temporal input.** The current model uses T=1 (post-fire image only). A T=2 (pre+post) variant raised in-domain IoU by +18.6% (Corrientes) and helped fine-tuned transfer (Cordoba 0.33 to 0.81), but *hurt* zero-shot transfer in the one case tested (Cordoba 0.115 to 0.087, see CHANGELOG v1.7). Not a safe assumption either way -- see the v3 redesign below.

**Chile threshold mismatch.** IoU=0.175 is well below the model's held-out validation number (0.651) because the fixed decision threshold (P>0.450, calibrated on Corrientes + Australia) is not precision-optimal for this biome. AUC-ROC=0.855 shows the model still ranks burned vs. unburned pixels well. A full threshold sweep against the dNBR reference (51 steps, 0.00-1.00) found the best possible IoU is 0.178, only 1.3% above the fixed threshold -- this gap is a discrimination/domain-shift problem, not a calibration problem, and no amount of re-thresholding closes it for this site.

**Adding a second training biome (Australia) did not uniformly help zero-shot transfer.** Cordoba, Greece, and Canada were only ever evaluated with the pre-Australia checkpoint; re-running all three with the current checkpoint shows a mixed result, not the uniform "+10-20 IoU" the Roadmap previously assumed:

| Site | Pre-Australia IoU | Current IoU (fixed t=0.45) | Current IoU (Otsu, unsupervised) | Pre-Australia AUC-ROC | Current AUC-ROC | GT positive rate |
|---|---|---|---|---|---|---|
| Cordoba | 0.115 | 0.105 | 0.104 | 0.738 | 0.666 | 7.2% |
| Greece | 0.234 | 0.245 | **0.257** | 0.652 | 0.668 | 23.3% |
| Canada | 0.191 | 0.289 | 0.295 | 0.606 | 0.646 | 38.5% |

Greece improved modestly and Cordoba regressed on both IoU and AUC-ROC, with no calibration escape (a full threshold sweep found no improvement over the fixed threshold). Canada's threshold-sweep "best" result was initially 0.3855 (IoU at t=0.00) -- checked against the site's ground-truth positive rate (0.385) and found to be a mathematical artifact: classifying every pixel as burned always produces an IoU exactly equal to the positive rate, regardless of model skill. At its real operating threshold, Canada's IoU (0.289) is still below that trivial all-positive baseline. One additional training biome is not a reliable lever for zero-shot generalization to a third, unrelated biome.

**Per-scene adaptive threshold (Otsu) recovers part of the calibration gap, without labels.** Instead of one fixed threshold (0.450) across every biome, a per-scene threshold chosen from the shape of the predicted-probability distribution alone (Otsu's method, fully unsupervised) gives Greece IoU=0.257 and Canada IoU=0.295, and correctly finds no gain for Cordoba. Unlike the labeled sweep, it never selects the degenerate all-positive threshold. This is the recommended operating strategy for any new region, since ground-truth labels are not available at real deployment time.

**Why not fix the label instead of the threshold.** RBR (Relativized Burn Ratio, Parks et al. 2014) is designed to be more comparable across vegetation types than a fixed dNBR threshold, and is a likely part of a real fix. It was not applied in this round: only the already-thresholded binary burn mask was retained for Cordoba, Greece, and Canada during patch extraction, not the continuous dNBR or pre-fire NBR needed to compute RBR. Recomputing it requires reprocessing the original Sentinel-2 scenes for all three sites -- scoped into the v3 redesign below.

**Other known issues:** 16x16 pixel block artifacts inherited from the ViT patch tokenizer (minor effect on binary IoU); ~160m boundary uncertainty on vector perimeters from non-overlapping patch extraction; water/nodata false positives in lake-heavy scenes (NDWI filter partially mitigates this).

---

## Roadmap

| Priority | Improvement | Expected gain | Status |
|---|---|---|---|
| 1 | Second training biome (Mediterranean 2021 or boreal 2019) | Heterogeneous by site, not a uniform gain (see Limitations) | **Done; re-verified against Cordoba/Greece/Canada, mixed result** |
| 2 | Per-scene adaptive threshold (Otsu, unsupervised, no labels required) | Recovers part of the calibration gap where one exists | **Done** |
| 3 | Multi-temporal input T=3 (matching Prithvi-EO-2.0 pretraining) | Uncertain in zero-shot: helped in-domain and fine-tuned, hurt zero-shot in the one case tested | Proposed, deferred (see v3 redesign below) |
| 4 | Test-Time Augmentation (flip H/V average) | +2-5 IoU | Planned |
| 5 | Vector output (GeoPackage burn scar polygons + NDVI) | Operational | **Done** |
| 6 | FastAPI inference endpoint (coordinates + date to mask) | Deployment | Planned (MLOps) |
| 7 | Morphological post-processing (remove isolated pixels) | Precision | Planned |
| 8 | Interactive Leaflet dashboard (GitHub Pages) | Portfolio / interpretability | **Done** |
| 9 | Chile dNBR ground truth alignment + quantitative ZS metrics | Validation | **Done -- revealed a threshold/domain mismatch, not shown on the dashboard until resolved** |
| 10 | California and Cerrado (Brazil) zero-shot sites | Cross-biome coverage | Planned |
| 11 | ESA WorldCover land cover context per polygon | Interpretability | **Done** |
| 12 | **v3 redesign**: RBR (Parks et al. 2014) instead of a fixed dNBR threshold; per-scene normalized input features instead of raw reflectance; adaptive calibration built into training | Targets the actual root causes found in this round, rather than more training data | Planned as a scoped follow-up (targeted late summer 2026) |

Full changelog with per-version metrics and figures: **[CHANGELOG.md](CHANGELOG.md)**.

---

## Repository Structure

```
wildfire-spread/
+-- CHANGELOG.md                         Full version history: metrics, figures, detailed results
+-- docs/
|   +-- index.html                       Interactive Leaflet dashboard (GitHub Pages, data embedded inline)
+-- notebooks/                          Organized by the release that introduced each notebook
|   +-- v1.0/                            Base pipeline (consolidated from 14 exploratory notebooks)
|   |   +-- 01_data_acquisition.ipynb    Sentinel-2 L2A (CDSE) + NASA FIRMS download
|   |   +-- 02_preprocessing.ipynb       Band stacking, patch extraction, dNBR labels
|   |   +-- 03_baseline.ipynb            U-Net ResNet34 training + diagnostic evaluation
|   |   +-- 04_prithvi_training.ipynb    Prithvi-EO-1.0-100M fine-tuning v1.0-v1.5 (Colab A100)
|   |   +-- 05_cordoba_data.ipynb        Cordoba test set acquisition and preprocessing
|   |   +-- 06_cordoba_evaluation.ipynb  Geographic generalization + few-shot adaptation (Colab A100)
|   |   +-- 07_inference_demo.ipynb      Single-patch inference demo (Colab)
|   +-- v1.6/
|   |   +-- 04b_prithvi_t2.ipynb         Siamese T=2 temporal fusion (Colab A100)
|   +-- v1.8/
|   |   +-- 09_greece_zs_evaluation.ipynb  Cross-continental ZS evaluation on Greece 2023 (Colab A100)
|   +-- v1.9/
|   |   +-- 08_prithvi_v2_training.ipynb   Prithvi-EO-2.0-300M fine-tuning (Colab A100)
|   |   +-- 10_canada_zs_evaluation.ipynb  ZS evaluation on Canada NWT 2023 + decision support (Colab A100)
|   +-- v2.0/
|   |   +-- 15_train_v22.ipynb           Training (Corrientes + Australia) + Chile ZS (Colab A100)
|   +-- v2.2/
|   |   +-- 16_chile_prob_export_v22.ipynb  Chile probability raster export for dNBR alignment (Colab T4)
|   +-- v2.3/
|       +-- 17_reeval_zs_sites_v22.ipynb    Cordoba/Greece/Canada re-evaluation + adaptive threshold (Colab T4)
+-- results/                             Figures referenced throughout README and CHANGELOG
+-- scripts/
|   +-- 00_prefire_download.py           Download pre-fire Sentinel-2 tiles for T=2 pairs
|   +-- 03b_paired_patches.py            Build aligned pre/post patch pairs
|   +-- 09_greece_download.py            Download Sentinel-2 L2A for Alexandroupolis 2023
|   +-- 10_greece_patches.py             JP2 to GeoTIFF, dNBR, patch extraction (Greece)
|   +-- 11_canada_pipeline.py            Download + JP2 to GeoTIFF + dNBR + patches (Canada, combined)
|   +-- 15_chile_download.py             Download Sentinel-2 L2A for Valparaiso 2023
|   +-- 16_chile_patches.py              JP2 to GeoTIFF, dNBR, patch extraction (Chile)
|   +-- 17-20_california_cerrado_*.py    California and Cerrado download/patches (planned sites)
|   +-- 21_run_zs_pipeline.py            Orchestrates download + patch extraction for all ZS sites
+-- models/
|   +-- best_prithvi_v22_burn_scar_wildfire.pth  Current checkpoint (Prithvi-EO-2.0-300M + FPN)
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

Notebooks v1.0/01-03 and v1.0/05 run locally on CPU (4-5 hours total, mostly data download).
Notebooks v1.0/04, v1.6/04b, v1.0/06, v1.9/08, v1.8/09, v1.9/10, and v2.0/15 require a GPU (Google Colab A100 recommended). v2.2/16 and v2.3/17 are inference-only and run fine on a Colab T4.
Scripts 11 and 21 run locally for the Canada and Chile pipelines respectively (download + patch extraction, several hours each).

---

## Data Sources

| Dataset | Provider | Access |
|---|---|---|
| Sentinel-2 L2A | ESA / Copernicus Data Space Ecosystem (CDSE) | Free, registration required |
| VIIRS SNPP active fire | NASA FIRMS | Free, API key required |
| ERA5 reanalysis (fire weather) | Copernicus Climate Data Store (CDS) | Free, registration required |
| ESA WorldCover 10m (2021) | Microsoft Planetary Computer | Free |

---

## References

- Jakubik, J. et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence. arXiv:2310.18660
- Prithvi-EO-1.0-100M: [ibm-nasa-geospatial/Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M)
- Prithvi-EO-2.0-300M: [ibm-nasa-geospatial/Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
- Key, C.H. and Benson, N.C. (2006). Landscape Assessment: Ground measure of severity. USDA Forest Service
- Parks, S.A., Dillon, G.K., Miller, C. (2014). A New Metric for Quantifying Burn Severity: The Relativized Burn Ratio. Remote Sensing, 6(3), 1827-1844
- terratorch: [github.com/IBM/terratorch](https://github.com/IBM/terratorch)

---

## License

MIT
