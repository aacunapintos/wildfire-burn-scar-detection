# Wildfire Burn Scar Detection with Prithvi-100M and Sentinel-2

Semantic segmentation of wildfire burn scars using the IBM/NASA Prithvi-100M geospatial foundation model fine-tuned on Sentinel-2 L2A imagery. Trained on the 2021-2022 Corrientes, Argentina fire season (~900,000 ha burned) and evaluated for geographic generalization on an unseen region (Cordoba, 2020).

## Key Results

| Model | Labels | Region | Pixel IoU | Recall | Precision | AUC-ROC |
|---|---|---|---|---|---|---|
| U-Net ResNet34 | FIRMS active fire | Corrientes | 0.013 | 7% | 14% | — |
| **Prithvi-100M + FPN** | **dNBR burn scar** | **Corrientes** | **0.43** | **64%** | **57%** | — |
| Prithvi-100M + FPN | dNBR | Cordoba (zero-shot) | 0.13 | 75% | 13% | 0.73 |
| **Prithvi-100M + FPN (few-shot FT)** | **dNBR** | **Cordoba (100 patches)** | **0.28** | **59%** | **34%** | **0.85** |

33x improvement over the FIRMS-based baseline. Few-shot fine-tuning of the decoder on 100 Cordoba patches improves IoU 2.14x over zero-shot transfer and raises AUC-ROC from 0.73 to 0.85, with the encoder kept frozen throughout.

![Portfolio overview](results/validation_overview.png)

*Best, median, and worst-performing patches from the Corrientes validation set (1,137 patches). Error maps: green = true positive, orange = false positive, red = false negative.*

## Approach

### The label problem

Initial training used NASA FIRMS active fire detections as ground truth. Only 2.6% of patches contained fire pixels, producing a pixel-level IoU of 0.013. The validation metric appeared higher (0.50) because empty patches scored 1.0 trivially, inflating the per-batch average.

FIRMS detects active fire (thermal anomaly), not burn scars. A pixel that burned three days ago leaves no thermal signal but remains a burned area. The correct label source is dNBR (differenced Normalized Burn Ratio), computed from pre- and post-fire Sentinel-2 imagery.

```
dNBR = NBR_pre - NBR_post     where NBR = (B8A - B12) / (B8A + B12)
Burn scar threshold: dNBR > 0.10
```

This change increased positive patch coverage from 2.6% to 55.8% (21x more training signal) and enabled meaningful learning.

### Model architecture

| Component | Details |
|---|---|
| Backbone | Prithvi-EO-1.0-100M (IBM/NASA) |
| Pretraining | Masked autoencoding on HLS (Harmonized Landsat-Sentinel) |
| Decoder | Feature Pyramid Network (FPN), trained from scratch |
| Encoder | Frozen during fine-tuning (100M parameters) |
| Input bands | B02, B03, B04, B8A, B11, B12 at 10 m resolution |
| Patch size | 224x224 px |
| Loss | DiceLoss + FocalLoss, fire class weight = 5.0 |

## Dataset

### Training: Corrientes, Argentina

| | |
|---|---|
| Region | Corrientes Province, NE Argentina (wetlands and grasslands) |
| Coordinates | 59.5W-56.0W / 29.0S-26.5S |
| Fire event | December 2021 - February 2022 (austral summer, extreme drought) |
| Scenes | 6 Sentinel-2 L2A tiles, 0% cloud cover |
| Patches | 5,687 x 224x224 px |
| Positive rate | 55.8% (dNBR > 0.10) |
| Source | Copernicus Data Space Ecosystem (CDSE) |

### Test: Cordoba, Argentina (unseen region)

| | |
|---|---|
| Region | Cordoba Province, central Argentina (Sierras Chicas, xerophytic scrubland) |
| Coordinates | 65.5W-62.5W / 33.0S-30.5S |
| Fire event | October-November 2020 |
| Patches | 6,634 x 224x224 px |
| Positive rate | 63.7% (dNBR > 0.10) |

The Cordoba set is a strict generalization test: different region, different biome, different year.

## Results

### Training curves

![Training curves](results/training_curves_prithvi_burn_scar.png)

### dNBR labels versus FIRMS detections

![dNBR vs FIRMS](results/dnbr_vs_firms_comparison.png)

Left: FIRMS active fire detections (sparse, misses most burned area). Right: dNBR-derived burn scar mask (complete, spatially consistent).

### Sample predictions, Corrientes validation set

![Predictions Corrientes](results/predictions_prithvi_burn_scar.png)

### Geographic generalization: Cordoba

![Cordoba predictions](results/cordoba_predictions.png)

| Metric | Corrientes (val) | Cordoba (zero-shot) |
|---|---|---|
| IoU | 0.42 | 0.13 |
| Recall | 0.73 | **0.75** |
| Precision | 0.50 | 0.13 |
| AUC-ROC | — | 0.73 |

The model retains high recall in Cordoba (75% of real burn scars detected) but precision drops due to spectral distribution shift between the Corrientes wetlands biome and the Cordoba mountain scrubland. AUC-ROC of 0.73 confirms the model learned transferable burn-scar spectral features.

### Few-shot domain adaptation: Cordoba

The FPN decoder was fine-tuned on 100 Cordoba patches (encoder kept frozen). The remaining 6,534 patches were held out as the test set.

![Fine-tuning curves](results/cordoba_finetune_curves.png)

![Fine-tuned predictions](results/cordoba_finetune_predictions.png)

| Metric | Zero-shot (base) | Few-shot FT (100 patches) | Change |
|---|---|---|---|
| IoU | 0.13 | **0.28** | +0.15 |
| Recall | 0.75 | 0.59 | -0.16 |
| Precision | 0.13 | **0.34** | +0.21 |
| AUC-ROC | 0.73 | **0.85** | +0.13 |

The fine-tuning trades some recall for a large precision gain. Overall IoU improves 2.14x. AUC-ROC reaches 0.85, indicating strong discriminative ability after adaptation. The encoder was never updated — all improvement comes from adapting the 2M-parameter decoder to the new biome.

## Limitations and Ongoing Improvements

The main limitation is biome-induced domain shift. The FPN decoder was trained on a single biome (Corrientes wetlands) and did not encounter the spectral characteristics of mountain xerophytic vegetation, causing over-prediction in Cordoba.

Ongoing improvements:

- **Multi-region training:** include Cordoba and a third biome in the training split to reduce domain gap from the start
- **Spectral augmentation:** random per-band scaling during training to reduce spectral memorization

## Repository Structure

```
wildfire-burn-scar/
├── notebooks/
│   ├── 01_data_acquisition.ipynb        Sentinel-2 L2A (CDSE) + NASA FIRMS download
│   ├── 02_preprocessing.ipynb           Band stacking, patch extraction, dNBR labels
│   ├── 03_baseline.ipynb                U-Net ResNet34 training + diagnostic evaluation
│   ├── 04_prithvi_training.ipynb        Prithvi-100M fine-tuning (Colab A100)
│   ├── 05_cordoba_data.ipynb            Córdoba test set acquisition and preprocessing
│   ├── 06_cordoba_evaluation.ipynb      Geographic generalization + few-shot adaptation (Colab A100)
│   └── 07_inference_demo.ipynb          Single-patch inference demo (Colab)
├── results/
│   ├── validation_overview.png          Best/median/worst patches, training curve, global metrics
│   ├── threshold_sweep.png              Metrics vs threshold + PR curve (v1.1)
│   ├── training_curves_prithvi_burn_scar.png
│   ├── predictions_prithvi_burn_scar.png
│   ├── dnbr_vs_firms_comparison.png
│   ├── cordoba_predictions.png
│   ├── cordoba_finetune_curves.png
│   └── cordoba_finetune_predictions.png
├── environment.yml
└── .gitignore
```

## Reproduce

**Environment**

```bash
conda env create -f environment.yml
conda activate geoai-wildfire
```

**Credentials**

Copy `.env.example` to `.env` and fill in your credentials:

```
CDSE_USER=your_copernicus_user
CDSE_PASSWORD=your_copernicus_password
FIRMS_API_KEY=your_firms_key
CDS_URL=https://cds.climate.copernicus.eu/api
CDS_KEY=your_cds_key
```

- CDSE: free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
- FIRMS: free API key at [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/api/area/)
- CDS: free account at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)

**Run order**

Notebooks 01-03 and 05 run locally on CPU (~4-5 hours total, mostly data download).
Notebooks 04, 06, and 07 require a GPU and are designed for Google Colab (A100 recommended).

## Data Sources

| Dataset | Provider | Access |
|---|---|---|
| Sentinel-2 L2A | ESA / Copernicus Data Space | Free, registration required |
| VIIRS SNPP active fire | NASA FIRMS | Free, API key required |
| ERA5 reanalysis | ECMWF / Copernicus CDS | Free, registration required |

## Changelog

| Version | Change | Val IoU | Val F1 | Notes |
|---|---|---|---|---|
| v1.0 | Base model, threshold=0.50 | 0.42 | 0.591 | Prithvi-100M + FPN, 40 epochs |
| v1.1 | Optimal threshold t=0.65 | **0.43** | **0.604** | Post-processing only, no retraining. Precision +15%, false positives reduced. |

## References

- Jakubik, J. et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence. arXiv:2310.18660.
- HuggingFace model: [ibm-nasa-geospatial/Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M)
- Key, C.H. and Benson, N.C. (2006). Landscape Assessment: Ground measure of severity. USDA Forest Service.
- terratorch: [github.com/IBM/terratorch](https://github.com/IBM/terratorch)
