# Meta-learning for cross-basin landslide susceptibility transfer

Code and figure-generation scripts for the manuscript
*"Meta-learning for cross-basin landslide susceptibility transfer across 14 Chilean watersheds: a remote sensing benchmark under spatial cross-validation"*
(Parra, Gil-Costa, Bonacic & Marín — under review, ISPRS Journal of Photogrammetry and Remote Sensing).

## What this repository contains

```
.
├── scripts/                  Full ML pipeline (numbered execution order)
│   ├── _meta_lib.py          MLP backbone + meta-learning utilities
│   ├── _cnn_lib.py           Spatial CNN backbone (U-Net-style baseline)
│   ├── 01-13 feature pipeline (SurtGIS factors → CFS-selected H5 datasets)
│   ├── 14 main benchmark     Reptile / FOMAML / Fine-tune / Independent / DANN
│   ├── 16 spatial benchmark  Spatial K-fold CV (replaces random sampling)
│   ├── 18 classical ML       Logistic regression, RF, XGBoost, CatBoost
│   ├── 23 extended baselines ProtoNet, Meta-Baseline, CDAN (5 seeds)
│   ├── 24 variogram          Empirical variogram + K-folds sensitivity
│   ├── 25 patch extraction   Multi-channel patches for the CNN baseline
│   ├── 26 U-Net benchmark    Spatial CNN vs MLP comparison on Huasco
│   └── ...                   Adaptation curves, ablations, climate distance, etc.
├── figures_pub/
│   ├── src/                  Publication figure scripts (matplotlib)
│   ├── utils/style.py        Shared style (Wong palette, ISPRS column widths)
│   └── out/                  Generated PDFs
├── paper/                    LaTeX source (Elsevier elsarticle template)
├── results/*/summary.csv     Aggregated benchmark results (means + bootstrap CIs)
└── data/aligned/cfs_selected_features.txt   The 11 CFS-selected features
```

Raw rasters, per-basin H5 feature tables, and intermediate patch tensors are
**not** included in the repository because of size. See "Reproducing the
results" below for how to regenerate them from public data sources.

## Reproducing the results

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn h5py joblib matplotlib geopandas rasterio
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

PyTorch CPU is sufficient; total benchmark wallclock is ~3 h on 8 vCPU.

### 2. Data sources (public)

| Layer                              | Source                                                |
|------------------------------------|-------------------------------------------------------|
| Basin polygons (BNA)               | DGA Chile (`Cuencas_BNA.shp`)                         |
| Landslide inventories              | SERNAGEOMIN *catastro de remociones en masa*; Serey et al. (2019) for Maule 2010; Parra (2025) thesis supplementary data for Chañaral and Taltal |
| DEM                                | MERIT-DEM (3 arc-sec)                                 |
| Optical                            | Sentinel-2 L2A (Copernicus, 2023 median composite)    |
| Climate                            | WorldClim 2.1 bioclimatic variables                   |
| Geology                            | SERNAGEOMIN 1:1,000,000 lithology map                 |
| Terrain factor computation         | SurtGIS toolkit, https://github.com/franciscoparrao/surtgis |

### 3. Pipeline

Run scripts in numerical order. Each script writes to `data/` or `results/`.
The main spatial benchmark (`16_spatial_benchmark.py`) is the entry point that
reproduces Table 1 of the manuscript. `23_extended_baselines.py` and
`24_variogram_validation.py` reproduce the supplementary baselines and the
variogram validation respectively. `26_unet_benchmark.py` reproduces the U-Net
vs MLP comparison on Huasco (Figure 6 of the manuscript).

### 4. Figures

```bash
cd figures_pub/src
for f in fig*.py figS*.py; do python3 "$f"; done
```

Outputs land in `figures_pub/out/`. The LaTeX source in `paper/` consumes those
PDFs via `\graphicspath`.

## Cite

If this code is useful in your own work, please cite the manuscript (Parra et
al., 2026, *ISPRS J. Photogramm. Remote Sens.*, under review).

## License

MIT (see `LICENSE`).

## Acknowledgments

Funded by the Postdoctoral Project DICYT 062619MC_POSTDOC, Universidad de
Santiago de Chile. SERNAGEOMIN for public access to the landslide *catastro*.
Map lines in any visualizations delineate study areas and do not necessarily
depict accepted national boundaries.
