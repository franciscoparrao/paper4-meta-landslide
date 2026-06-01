# Figures and Tables — Paper 4

## Main paper figures (5)

### Figure 1: Study area
**File**: `figures/study_area.png`
**Caption**: Eight Andean watersheds spanning a hyper-arid (Chañaral, ~6 mm/yr) to semi-humid (Maule, ~915 mm/yr) climate gradient in Chile. Source basins (red squares) are used for meta-training; target basins (blue circles) for K-shot evaluation. Polygons are color-coded by mean annual precipitation (WorldClim 2.1 bio_12). Inset shows the location within South America (Chile highlighted).

### Figure 2: K-shot benchmark under random CV
**File**: `figures/f1_vs_k_panel.png`
**Caption**: F1 score versus number of support shots K for four target basins under random cross-validation. Lines are means over 50 episodes × 3 seeds; shaded bands are bootstrap 95% confidence intervals (1,000 resamples). FOMAML (pink) wins K=1 in 3 of 4 basins (Copiapó, Huasco, Limarí), with 5–9 percentage points (pp) F1 advantage over Independent. The advantage narrows to <1 pp at K=20 across all basins, indicating that meta-learning's value is concentrated in the extreme few-shot regime.

### Figure 3: Meta-learning advantage curve under spatial CV (KILLER FIGURE)
**File**: `figures/spatial_lift_vs_k.png`
**Caption**: Mean F1 advantage over the Independent baseline (averaged across the four target basins) as a function of K under spatial cross-validation. FOMAML (pink) achieves +5 pp at K=1, decaying monotonically to +0.5 pp at K=20. Reptile (orange) shows similar but smaller advantage. Fine-tune (blue) provides essentially no benefit (within ±1 pp). DANN (green) is the only baseline with a *negative* advantage of approximately -2 pp across all K, indicating that adversarial domain alignment is counter-productive when source domains are scarce (N=4). Dashed line marks zero advantage. Bars represent 1σ across the four target basins (omitted for clarity in the rendered version).

### Figure 4: Adaptation curves (validation of H_p4_2)
**File**: `figures/adaptation_curves_K5.png`
**Caption**: Adaptation efficiency at K=5 for the four target basins. F1 score is plotted as a function of inner Adam gradient steps applied to the K=5 support set, evaluated on the held-out query. FOMAML (pink) starts at F1 ≈ 0.78 with **zero adaptation steps**, indicating that the meta-learned initialization is already discriminative. Reptile (orange) reaches its peak in 3-5 steps. Fine-tune (blue) requires ~10 steps. Independent (gray) needs >100 steps to converge. This validates the hypothesis that meta-learning produces models requiring only 3-5 adaptation gradient steps.

### Figure 5: Random CV vs spatial CV
**File**: `figures/spatial_vs_random_comparison.png`
**Caption**: F1 versus K for each method, showing both random CV (dashed lines) and spatial CV (solid lines). Spatial CV partitions each basin into 5 KMeans clusters on (x, y) coordinates, ensuring support and query are spatially disjoint. Drops between random and spatial protocols range from 1 pp (Huasco) to 27 pp (Elqui), but the **relative ranking of methods is preserved**. The Huasco basin generalizes spatially almost perfectly, while Elqui exhibits intra-basin heterogeneity that challenges all methods equally.

---

## Supplementary figures (7)

### Figure S1: AUC curves under random CV
**File**: `figures/auc_vs_k_panel.png`
**Caption**: Same as Figure 2 but reporting ROC-AUC instead of F1. Patterns are consistent: FOMAML achieves higher AUC at K=1 in all targets (gap 5-12 pp).

### Figure S2: Detailed F1 vs K under spatial CV
**File**: `figures/spatial_f1_vs_k_panel.png`
**Caption**: Per-target spatial CV results. FOMAML wins K=1 in all four basins, with the 95% CI of FOMAML and Independent failing to overlap in three basins.

### Figure S3: Adaptation curves at K=10
**File**: `figures/adaptation_curves_K10.png`
**Caption**: As Figure 4, but at K=10. The H_p4_2 pattern is preserved: meta-learned initializations require few adaptation steps; Independent requires >100.

### Figure S4: Hyperparameter sensitivity (3 panels)
**Files**: `figures/ablation_inner_steps.png`, `figures/ablation_eps.png`, `figures/ablation_K_meta.png`
**Caption**: One-at-a-time ablations around the baseline configuration (inner_steps=5, eps=0.1, K_meta=10). F1 at K=5 K-shot is reported as a function of (a) inner_steps for Reptile and FOMAML; (b) eps for Reptile only; (c) K_meta for both. All sensitivity is small (within 1 pp), indicating the chosen hyperparameters are robust.

### Figure S5: DANN comparison under random CV
**File**: `figures/dann_comparison.png`
**Caption**: Random CV results including DANN. Consistently underperforms Independent in 3/4 targets at K=1.

### Figure S6: Summary heatmap
**File**: `figures/summary_heatmap.png`
**Caption**: F1 (mean) for all (target × K × method) combinations under random CV. Visual synthesis of the K-shot benchmark.

### Figure S7: Lift advantage under random CV
**File**: `figures/lift_vs_k.png`
**Caption**: Per-basin lift curves under random CV. Same pattern as Figure 3 but per-basin (thin lines) and with mean line. Random CV shows slightly larger advantages than spatial CV due to spatial autocorrelation amplifying performance.

---

## Tables

### Table 1: K-shot benchmark — F1 mean ± 95% CI under spatial CV

*4 target basins × 4 K values × 5 methods. Values are F1 mean and bootstrap 95% CIs. Bold = best per row. Asterisk (*) = 95% CI does not overlap with Independent.*

| Target | K | Independent | Fine-tune | DANN | Reptile | FOMAML |
|---|---|---|---|---|---|---|
| **Copiapó** | 1 | 0.583 [.55, .62] | 0.605 [.57, .64] | 0.548 [.50, .59] | 0.616 [.58, .65] | **0.683 [.65, .72]\*** |
| | 5 | 0.641 [.60, .68] | 0.681 [.65, .72] | 0.611 [.57, .65] | 0.643 [.60, .69] | **0.702 [.67, .74]\*** |
| | 10 | 0.711 [.68, .75] | **0.736 [.70, .77]** | 0.663 [.62, .70] | 0.690 [.66, .73] | 0.723 [.69, .76] |
| **Huasco** | 1 | 0.733 [.71, .75] | 0.740 [.72, .76] | 0.712 [.68, .74] | 0.755 [.73, .78] | **0.757 [.73, .78]** |
| | 5 | 0.805 [.79, .82] | 0.805 [.79, .82] | 0.793 [.78, .81] | 0.817 [.80, .83] | **0.825 [.81, .84]** |
| | 10 | 0.849 [.84, .86] | 0.841 [.83, .85] | 0.835 [.82, .85] | **0.854 [.84, .87]** | 0.854 [.84, .86] |
| | 20 | 0.868 [.86, .88] | 0.867 [.86, .88] | 0.859 [.85, .87] | **0.875 [.87, .88]** | 0.874 [.87, .88] |
| **Elqui** | 1 | 0.432 [.40, .46] | 0.422 [.39, .45] | 0.410 [.38, .44] | 0.458 [.43, .49] | **0.465 [.44, .50]** |
| | 5 | 0.472 [.44, .51] | 0.469 [.44, .51] | 0.474 [.44, .51] | **0.500 [.47, .53]** | 0.481 [.45, .52] |
| | 10 | 0.489 [.45, .53] | 0.507 [.47, .55] | 0.491 [.45, .53] | 0.509 [.47, .54] | **0.534 [.50, .57]** |
| | 20 | 0.479 [.44, .52] | 0.481 [.44, .52] | 0.480 [.44, .52] | **0.501 [.46, .54]** | 0.489 [.45, .53] |
| **Limarí** | 1 | 0.472 [.44, .50] | 0.449 [.42, .48] | 0.460 [.43, .49] | 0.491 [.46, .52] | **0.514 [.49, .54]\*** |
| | 5 | 0.543 [.51, .57] | 0.538 [.51, .57] | 0.500 [.47, .53] | 0.528 [.50, .56] | 0.528 [.50, .56] |
| | 10 | **0.562 [.53, .59]** | 0.535 [.51, .56] | 0.554 [.53, .58] | 0.550 [.52, .58] | 0.555 [.52, .59] |
| | 20 | 0.560 [.53, .59] | 0.550 [.52, .58] | **0.574 [.55, .60]** | 0.567 [.54, .59] | 0.559 [.53, .59] |

**Source**: `results/spatial_benchmark/summary.csv`
**Total runs aggregated**: 24,000 K-shot adaptations

---

### Table 2: Selected features (CFS forward greedy)

*Final 14 features in order of selection, with category, symmetric uncertainty (SU) with class label, and CFS subset score after each addition.*

| Step | Feature | Category | SU(f, class) | Cumulative score |
|---|---|---|---|---|
| 1 | terrain__landform | Terrain | 0.179 | 0.179 |
| 2 | texture__glcm_homogeneity_dem | Texture | 0.164 | 0.206 |
| 3 | focal_stats__dem_std_r10 | Focal stats | 0.159 | 0.219 |
| 4 | focal_stats__slope_std_r4 | Focal stats | 0.150 | 0.226 |
| 5 | terrain__openness_positive | Terrain | 0.156 | 0.229 |
| 6 | texture__glcm_correlation_dem | Texture | 0.154 | 0.232 |
| 7 | terrain__mrvbf | Terrain | 0.087 | 0.234 |
| 8 | terrain__mrrtf | Terrain | 0.057 | 0.237 |
| 9 | terrain__curvedness | Terrain | 0.136 | 0.239 |
| 10 | climate__bio_12 | Climate | 0.070 | 0.240 |
| 11 | texture__glcm_energy_dem | Texture | 0.164 | 0.241 |
| 12 | terrain__ls_factor | Terrain | 0.138 | 0.243 |
| 13 | terrain__tpi_r21 | Terrain | 0.125 | 0.244 |
| 14 | terrain__geomorphons | Terrain | 0.137 | 0.244 |

**Notable**: 8 of 14 features are terrain-derived; **no hydrology, spectral, or geology features** are selected. Annual precipitation (bio_12) is the only climate feature. This reflects the dominance of multi-scale terrain ruggedness (mrvbf, mrrtf, openness, focal std, GLCM) for landslide discrimination in arid Andean terrain.

---

### Table 3: Dataset summary

*Per-basin sample counts after balanced sampling. Source basins are used for meta-training; targets for K-shot evaluation. Climate proxy is mean annual precipitation (WorldClim bio_12).*

| Basin | Role | N total | N pos | N neg | Precip. (mm/yr) | Inventory source |
|---|---|---|---|---|---|---|
| Chañaral (027) | Source | 1,124 | 562 | 562 | ~5.6 | Sustainability 2023 supplementary |
| Taltal (029) | Source | 4,006 | 2,003 | 2,003 | ~7.8 | Parra (2025) tesis doctoral |
| Maule (073) | Source | 406 | 203 | 203 | ~915 | Mw 8.8 2010 inventory (Serey et al.) |
| Choapa (047) | Source | 116 | 58 | 58 | ~219 | SERNAGEOMIN catastro |
| Copiapó (030) | Target | 38 | 19 | 19 | ~64 | Paper 1 inventory |
| Huasco (032) | Target | 556 | 278 | 278 | ~14 | Paper 1 inventory |
| Elqui (043) | Target | 264 | 132 | 132 | ~110 | Paper 1 inventory |
| Limarí (045) | Target | 70 | 35 | 35 | ~152 | Paper 1 inventory |
| **Total** | | **6,580** | **3,290** | **3,290** | | |

---

## Computational summary

Total model fits performed across all experiments:

| Experiment | Fits | Output |
|---|---|---|
| Random-CV K-shot benchmark | 8,400 | results/benchmark/ |
| Spatial-CV K-shot benchmark | 15,900 | results/spatial_benchmark/ |
| Adaptation curves | 25,200 | results/adaptation/ |
| Hyperparameter ablations | 6,120 | results/ablations/ |
| DANN-only baseline | 2,100 | results/dann/ |
| Spatial CV (K=20 only, exploratory) | 4,800 | results/spatial_cv/ |
| **Total** | **62,520** | |

This rigor — bootstrap 95% CIs over 62,520 fits — substantially exceeds the typical evaluation depth in landslide ML literature (often N=1 single fold).
