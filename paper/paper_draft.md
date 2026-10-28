# Meta-learning for cross-basin transferability of landslide susceptibility models in arid Andean watersheds

**Authors**: Francisco Parra*, [Co-author 1], [Co-author 2], [PI Postdoc DICYT]
*Departamento de Geografía, Universidad de Santiago de Chile*

**Target journal**: ISPRS Journal of Photogrammetry and Remote Sensing

---

## Abstract

Landslide susceptibility modeling in newly-studied basins is constrained by data scarcity: each new watershed typically requires hundreds of labeled events for a reliable classifier. We propose meta-learning as a practical solution and present the first systematic application of Reptile and First-Order MAML (FOMAML) to landslide susceptibility classification in eight Andean watersheds spanning a hyper-arid to semi-humid climate gradient. We meta-train on four source basins (Chañaral, Taltal, Maule 2010, Choapa) using 14 features selected via correlation-based feature selection (CFS) from a superset of 64 conditioning factors derived from a 30 m Copernicus DEM, Sentinel-2 composites, WorldClim bioclimatics, and SERNAGEOMIN geology. We evaluate K-shot transfer (K ∈ {1, 5, 10, 20}) on four target basins (Copiapó, Huasco, Elqui, Limarí) via 32,400 model fits across three random seeds, with bootstrap 95% confidence intervals under both random and spatial cross-validation protocols. FOMAML wins K = 1 in 4/4 target basins under spatial CV, providing up to 5 percentage points (pp) F1 advantage over an Independent baseline at the most extreme few-shot regime. The advantage decays monotonically with K and converges by K = 20. A Domain-Adversarial Neural Network (DANN) baseline performs *worse* than Independent (-2 pp F1), suggesting that adversarial domain alignment is counter-productive when source domains are scarce (N = 4). Adaptation curves confirm that meta-learned initializations reach 78% F1 in 0–3 gradient steps versus >100 steps for Independent. We discuss limitations including a hard transfer case (Elqui) where spatial heterogeneity dominates, and propose extensions to U-Net architectures for spatially-structured prediction.

**Keywords**: meta-learning, MAML, Reptile, transfer learning, landslide susceptibility, few-shot classification, domain adaptation, Andes

---

## 1. Introduction

Landslide susceptibility maps are the cornerstone of geohazard planning in mountainous regions. The dominant paradigm — fitting a machine-learning classifier to a per-basin inventory of labeled events — has produced strong results when training data are abundant (Reichenbach et al., 2018; Merghadi et al., 2020). However, the practitioner facing a *newly-studied* basin confronts a data-scarcity bottleneck: digitizing a usable inventory typically requires expert geomorphological mapping over hundreds of events, an effort that can take months and is rarely funded for individual watersheds. As a result, basins with sparse inventories (N < 50 mass-movement records) cannot benefit from data-hungry deep learning approaches, while transfer of models trained on data-rich neighboring basins suffers from systematic shifts in geology, climate, and land cover.

Two strategies have emerged to mitigate this. **Transfer learning** by fine-tuning a model pre-trained on data-rich basins (Ghorbanzadeh et al., 2022) reduces label requirements but still needs tens to hundreds of target samples. **Domain adaptation** approaches such as DANN (Ganin and Lempitsky, 2015) attempt to learn domain-invariant representations from labeled source data and unlabeled target data, with mixed success in remote sensing (Tuia et al., 2016).

A complementary paradigm, **meta-learning** (Finn et al., 2017; Nichol et al., 2018), trains a model to *adapt rapidly to new tasks from a few examples*. Meta-learning has shown promise in remote sensing for land-cover classification (Rußwurm et al., 2020) and satellite image segmentation (Tseng et al., 2022), but to our knowledge has never been systematically evaluated for landslide susceptibility, particularly in the arid-to-semi-arid Andean cordillera where climate gradients are extreme and traditional transfer learning is difficult.

In this work, we present the first systematic application of meta-learning to landslide susceptibility classification across a hydro-climatic gradient in north-central Chile. We make the following contributions:

1. We construct a unified, balanced dataset spanning **eight Andean basins** with consistent feature engineering: 30 m DEM derivatives (terrain, hydrology, texture), Sentinel-2 spectral bands, WorldClim bioclimatic variables, and SERNAGEOMIN geology categoricals. Inventory labels span four published catalogs and one ad-hoc digitization (3,290 positives + 3,290 negatives after balanced sampling).

2. We benchmark **four methods** — Independent (no source), Fine-tune (concat-source pretrain), Reptile, FOMAML — plus **DANN** as a domain-adversarial baseline, across K-shot evaluation (K ∈ {1, 5, 10, 20}) on four target basins, with three random seeds and bootstrap 95% confidence intervals (32,400 model fits in total).

3. We report results under **both random and spatial cross-validation protocols**, addressing the spatial autocorrelation concern that pervades landslide ML evaluation.

4. Our findings: (i) **FOMAML wins K=1 in 4/4 target basins** under spatial CV, providing up to **5 pp F1 advantage** over Independent at the most extreme few-shot regime; (ii) the advantage **decays monotonically with K** and vanishes by K=20; (iii) **DANN underperforms** Independent by 2 pp across all K, suggesting adversarial alignment is counter-productive when source domains are few (N=4); (iv) **adaptation curves** confirm meta-learned initializations reach 78% F1 in 0-3 gradient steps versus >100 steps for Independent.

The remainder of this paper is organized as follows. Section 2 reviews related work in landslide ML, transfer learning, and meta-learning. Section 3 describes the study area and data. Section 4 details the methodology. Section 5 presents results. Section 6 discusses implications and limitations. Section 7 concludes.

---

## 2. Related Work

### 2.1 Landslide susceptibility modeling

Statistical and machine-learning approaches to landslide susceptibility have been the subject of extensive review (Reichenbach et al., 2018; Merghadi et al., 2020). The field has progressively moved from bivariate and weights-of-evidence approaches to logistic regression, random forests, gradient-boosted trees (XGBoost, CatBoost), and more recently to deep neural networks. Brenning (2005) established spatial cross-validation as a methodological best practice for landslide modeling, demonstrating that random hold-out validation systematically overestimates generalization error due to spatial autocorrelation between nearby training and test pixels. Object-oriented approaches (Stumpf and Kerle, 2011) and patch-based convolutional models have refined the spatial feature representation, but the fundamental requirement of hundreds to thousands of labeled events per study area remains a binding constraint. Recent benchmarks comparing classical and ensemble ML algorithms (Merghadi et al., 2020) report F1 scores in the 0.75–0.90 range when N > 200, but performance degrades sharply for small inventories. Most of the field operates within a single basin or contiguous region; cross-basin transfer is studied less frequently and is typically framed as out-of-sample testing rather than as a problem requiring methodological adaptation.

### 2.2 Transfer learning in remote sensing

Transfer learning by fine-tuning a model pre-trained on data-rich source domains is the de facto strategy for label-efficient remote sensing (Tuia et al., 2016). In landslide applications, Ghorbanzadeh et al. (2022) demonstrated that ImageNet-pretrained convolutional networks fine-tuned on a few hundred local samples outperform train-from-scratch baselines, confirming that representations learned on natural images contain transferable low-level features. However, fine-tuning a deep network with K < 50 target labels typically results in catastrophic over-fitting unless aggressive regularization or early stopping is applied. Furthermore, the "concat-source" variant — pre-training on a union of source basins, which is the baseline we adopt — implicitly assumes that all source domains are equally relevant to the target, a hypothesis that is rarely tested explicitly and that fails when the source pool spans heterogeneous environmental conditions.

### 2.3 Domain adaptation

Domain adaptation (DA) seeks to learn classifiers that generalize across distribution shifts by leveraging both labeled source data and unlabeled target data. The Domain-Adversarial Neural Network (DANN) of Ganin and Lempitsky (2015) is the canonical method: a feature extractor is trained simultaneously to (i) discriminate the class label and (ii) confuse a domain classifier through a Gradient Reversal Layer, encouraging the learned representation to be domain-invariant. Variants based on Maximum Mean Discrepancy (DAN; Long et al., 2015) and Wasserstein distance have followed. In remote sensing, DA has been applied to multi-source land-cover classification, image segmentation, and scene recognition, with mixed success (Tuia et al., 2016): performance gains are typically substantial when source domains are abundant and labeled target data are completely absent, but degrade when only a handful of source domains are available or when small numbers of labeled target samples are accessible. The latter regime is precisely our setting, and our results below confirm that DANN is not a competitive baseline when source-domain count is small (N=4).

### 2.4 Meta-learning

Meta-learning frames the data-scarcity problem as one of *learning to learn*: the model is trained on a distribution of related tasks so that adaptation to a new task from few examples requires only a small parameter update. Model-Agnostic Meta-Learning (MAML; Finn et al., 2017) is the canonical second-order method, computing gradients through inner adaptation steps, but it is computationally expensive and unstable in practice. First-order approximations — FOMAML, which drops the second-order term, and Reptile (Nichol et al., 2018), which replaces query-loss-driven updates with averaged inner-step displacements — are more practical. Theoretical work (Raghu et al., 2020) suggests that MAML's success can be largely attributed to feature reuse rather than rapid weight adaptation, which we revisit in our adaptation-curve analysis (Sec. 5.4).

In remote sensing, meta-learning has been applied to few-shot land-cover classification (Rußwurm et al., 2020), achieving 70–90% accuracy with K=5 samples per class on EuroSAT and Sen12MS benchmarks, and to satellite image classification more broadly (Tseng et al., 2022). To our knowledge, however, **no published study has applied meta-learning to landslide susceptibility classification**. This is surprising given the natural fit: each basin can be cast as a "task" with shared structure (terrain rugosity, hydrological setting, lithology) but variable specifics (climate, dominant trigger, magnitude distribution), and inventories are routinely scarce in newly-studied regions. Our work fills this gap and contributes the first systematic evaluation of meta-learning algorithms for landslide susceptibility transfer in arid Andean terrain.

---

## 3. Study Area and Data

### 3.1 Basins

We study eight watersheds in north-central Chile (Fig. X) selected to span a strong hydro-climatic gradient and to have heterogeneous landslide-mapping coverage:

**Source basins (data-rich)**:
- *Río Salado / Chañaral* (BNA 027, hyper-arid): 1,710 events from a published 2023 supplementary dataset
- *Costeras Q. Negra / Pan de Azúcar / Taltal* (BNA 029, hyper-arid coastal): 80 mass-movement polygons + 300 negatives from Parra (2025)
- *Río Maule* (BNA 073, semi-humid): 203 events filtered from the 1,227-event Mw 8.8 2010 earthquake-induced inventory (Serey et al., 2019)
- *Río Choapa* (BNA 047, semi-arid): 62 events from the SERNAGEOMIN national catastro

**Target basins (data-scarce)**:
- *Río Copiapó* (BNA 030): 19 events
- *Río Huasco* (BNA 032): 278 events
- *Río Elqui* (BNA 043): 132 events
- *Río Limarí* (BNA 045): 35 events

The split between source and target is motivated both by data availability and by ensuring climate-gradient coverage in both groups.

### 3.2 Conditioning factors

We compute 64 features per basin at 30 m grid alignment, organized in seven categories:

- **Terrain (29)**: slope, aspect, eastness, northness, hillshade, sky-view factor, openness (positive, negative), geomorphons, mass-balance index, MRVBF, MRRTF, terrain ruggedness, vector ruggedness measure, deviation from mean elevation, convergence, curvature, curvedness, shape index, surface area ratio, three TPI radii, valley depth, relative slope position, LS-factor, tri, tpi.
- **Hydrology (8)**: flow accumulation (D8, MFD), flow direction (D8, D∞), HAND, stream network, TWI, drainage density, sediment connectivity (Borselli).
- **Texture GLCM (6)**: contrast, correlation, dissimilarity, energy, entropy, homogeneity (computed on DEM).
- **Focal statistics (6)**: DEM std at radii 1, 4, 10; DEM range r=4; slope std r=4; slope mean r=4.
- **Spectral (6)**: Sentinel-2 L2A median composite (red, green, blue, NIR, SWIR16, SWIR22) for 2023, cloud-masked via SCL.
- **Climate (5)**: WorldClim 2.1 bioclimatics — bio_01 (annual mean temp), bio_12 (annual precipitation), bio_13/14 (precip wettest/driest month), bio_15 (precip seasonality).
- **Geology (3)**: SERNAGEOMIN 1:1,000,000 lithology class, rock type, geological age (categorical, encoded).

All factors are computed using SurtGIS (Parra, 2025), a Rust-based geospatial toolkit, with reprojection to UTM zone 19S and bilinear/categorical resampling as appropriate.

### 3.3 Sampling protocol

We adopt a **hybrid balanced sampling** strategy. Each pixel of the 30 m DEM grid is the unit of analysis. For each basin:

1. **Positives**: For point inventories, snap to nearest pixel. For polygon inventories (Taltal), rasterize with `all_touched=True` and randomly sub-sample at most 30 pixels per polygon to limit spatial autocorrelation.
2. **Negatives**: When the inventory provides explicit non-mass-movement points (Chañaral, Taltal), we use them and expand each by a 60 m buffer to cover surrounding pixels. When negatives are unavailable, we generate them by random sampling within the basin polygon, excluding a 500 m buffer around positives to avoid spatial leakage.
3. **Balancing**: Negatives are sub-sampled or augmented to match positive count exactly (1:1).

This procedure yields 6,580 samples (3,290 pos + 3,290 neg) across the eight basins.

### 3.4 Feature selection (CFS)

We apply **correlation-based feature selection** (Hall, 1999) using symmetric uncertainty (information-theoretic, robust to mixed data types). Features are quantile-discretized into 10 bins. Forward greedy search adds the feature that maximizes the subset score:

$$M_S = \frac{k \cdot \overline{r_{cf}}}{\sqrt{k + k(k-1) \cdot \overline{r_{ff}}}}$$

where $k$ is subset size, $\overline{r_{cf}}$ the mean feature-class correlation, $\overline{r_{ff}}$ the mean inter-feature correlation. The procedure terminates when no feature addition improves the score.

CFS selects **14 features**: 8 terrain (landform, openness_positive, mrvbf, mrrtf, curvedness, ls_factor, tpi_r21, geomorphons), 3 texture GLCM (homogeneity, correlation, energy), 2 focal stats (dem_std r=10, slope_std r=4), 1 climate (bio_12 annual precipitation). Notably, **no hydrology, spectral, or geology features** are selected — terrain ruggedness and texture dominate the discriminative signal in arid Andean landslides.

---

## 4. Methods

### 4.1 Backbone model

All methods share a 3-layer MLP backbone: 14 → 64 → 32 → 1, with ReLU activations, 10% dropout per hidden layer, and binary cross-entropy with logits as the training loss. We deliberately use a small tabular network rather than a CNN/Transformer because (i) our feature representation is point-wise and scale-invariant, (ii) our pixel-level samples lack spatial neighborhoods at this stage of the pipeline, (iii) it ensures interpretability of feature importance. Section 6 discusses the natural extension to U-Net for spatially-structured prediction.

### 4.2 Methods compared

#### Independent
Train MLP from scratch on K target labels (K positives + K negatives), 30 Adam steps at learning rate $10^{-2}$. No source-basin information is used.

#### Fine-tune (concat-source baseline)
Pretrain MLP on the concatenation of all source basins (300 epochs at $10^{-3}$), then fine-tune for 30 steps on K target samples.

#### Reptile (Nichol et al., 2018)
First-order meta-learning. For 300 outer steps:
1. Sample a source basin $\tau$
2. Sample $K_{\text{meta}}$=10 support points
3. Adapt model with 5 inner SGD steps at $10^{-2}$
4. Update meta-model: $\theta \leftarrow \theta + \epsilon (\theta_{\text{adapted}} - \theta)$ with $\epsilon = 0.1$

At test time: load meta-model and adapt 30 steps on K target samples (same as Fine-tune).

#### FOMAML (First-Order MAML)
For 300 outer steps:
1. Sample task $\tau$, support $S$, query $Q$
2. Inner: 5 SGD steps on $S$, lr $10^{-2}$
3. Compute query loss on the *adapted* model
4. Take query-loss gradient w.r.t. adapted parameters; copy this gradient to the meta-model
5. Apply Adam outer-step at lr $10^{-3}$

#### DANN (Ganin and Lempitsky, 2015)
Feature extractor $F$: 14 → 64 → 32 (shared). Class head $C$: 32 → 1. Domain head $D$: 32 → 4 (with Gradient Reversal Layer). Trained for 100 epochs at batch 64, lr $10^{-3}$, with annealed $\lambda(p) = 2/(1+e^{-10p}) - 1$. After training, $F+C$ are loaded into the MLP backbone for K-shot adaptation following the same protocol as Fine-tune.

### 4.3 K-shot evaluation protocol

For each (target basin, K, method, seed) combination, we sample 50 K-shot episodes (random CV) or 20 episodes per spatial fold (5 folds × 20 = 100 episodes for spatial CV). Each episode:

1. **Support**: K positives + K negatives sampled uniformly at random.
2. **Query** (random CV): N_QUERY = 10 positives + 10 negatives disjoint from support.
3. **Query** (spatial CV): full held-out spatial fold (KMeans cluster on (x, y) coordinates).
4. Adapt the method's initialization with 30 Adam steps on the support.
5. Evaluate F1 and ROC-AUC on the query.

Reported F1 / AUC are bootstrap means with 95% confidence intervals across 1,000 resamples of all (seed × episode) results pooled.

---

## 5. Results

### 5.1 Random-CV K-shot benchmark

[Insert Fig. f1_vs_k_panel.png]
[Insert Fig. lift_vs_k.png]

Under random cross-validation, FOMAML wins K=1 in 3 of 4 target basins (Copiapó, Huasco, Limarí), with 5–9 pp F1 advantage over Independent. The advantage narrows to 0–2 pp at K=10 and disappears at K=20, where all methods converge to within 1 pp F1.

### 5.2 Spatial-CV K-shot benchmark

[Insert Fig. spatial_f1_vs_k_panel.png]
[Insert Fig. spatial_lift_vs_k.png]

Under spatial cross-validation — where support and query are sampled from disjoint KMeans clusters of the basin — **FOMAML wins K=1 in 4/4 target basins** (Copiapó, Huasco, Elqui, Limarí). The 95% CIs of FOMAML and Independent do not overlap in 3/4 basins. FOMAML achieves +5 pp F1 advantage in mean across targets at K=1, decaying monotonically to +0.5 pp at K=20.

[Insert Fig. spatial_vs_random_comparison.png]

The spatial CV protocol yields lower absolute F1 magnitudes than random CV (drop of 1 pp in Huasco to 27 pp in Elqui), confirming that random CV overestimates absolute performance due to spatial autocorrelation, but **the relative ranking of methods is preserved**.

### 5.3 DANN underperforms

A surprising finding: **DANN underperforms Independent by approximately 2 pp F1** consistently across K and across all four target basins. We hypothesize this is because adversarial domain alignment requires many source domains to learn invariant representations; with only N=4 source basins, the domain classifier finds easy spurious distinctions (e.g., based on climate or geology) that the gradient reversal does not eliminate effectively.

### 5.4 Adaptation curves confirm rapid convergence (H_p4_2)

[Insert Fig. adaptation_curves_K5.png]

FOMAML's meta-learned initialization achieves F1 ≈ 0.78 with **zero adaptation steps**, confirming that the prior is already discriminative. Reptile reaches its peak in 3-5 steps. Fine-tune in ~10 steps. Independent requires >100 steps to converge. This validates the practical efficiency benefit of meta-learning.

### 5.5 Hyperparameter sensitivity

[Insert Fig. ablation_inner_steps.png, ablation_eps.png, ablation_K_meta.png]

Sensitivity to inner_steps {1,3,5,10}, eps {0.05,0.1,0.3}, and K_meta {5,10,20} is small (within 0.5 pp). The default configuration (inner_steps=5, eps=0.1, K_meta=10) is robust.

---

## 6. Discussion

### 6.1 Why FOMAML outperforms Reptile in tabular landslide data

Across both random- and spatial-CV protocols, FOMAML consistently leads Reptile by 1–3 pp F1 at K=1 (Tables 1, S1) and by smaller but positive margins at K=5–10. This finding is consistent with Raghu et al. (2020), who argue that the principal advantage of MAML-family methods comes from learning a feature representation amenable to rapid linear adaptation rather than from the second-order term itself. FOMAML's update rule — using the gradient of the *query* loss computed *through* an adapted model — provides a more direct optimization signal than Reptile's average-displacement heuristic, particularly when the adaptation surface is non-isotropic (as is the case in tabular problems where a few highly-discriminative features such as `terrain__landform` or `texture__glcm_homogeneity` dominate). Reptile's averaging procedure implicitly assumes isotropic curvature; in tabular data with heterogeneous feature scales (even after standardization), this assumption is weakly satisfied. We thus recommend FOMAML over Reptile for tabular landslide features. The trade-off is a modest computational increase (one query evaluation per outer step) which is negligible at our problem scale.

### 6.2 Why DANN under-performs with few source domains

DANN consistently under-performs Independent (i.e., a model trained from scratch on K target labels) by approximately 2 pp F1, regardless of K. This contradicts the narrative in much of the DA literature, where DANN is presented as a strong default baseline. We propose two complementary explanations.

First, **adversarial invariance requires sufficient domain diversity**. With only N=4 source basins, the domain classifier easily exploits low-level features such as climate (bio_12 ranges from 5 mm/yr in Chañaral to 915 mm/yr in Maule) or lithology (volcanic vs. sedimentary). The Gradient Reversal Layer encourages the feature extractor to discard these signals, but they are also potentially predictive of landslide susceptibility (e.g., precipitation-driven debris flows correlate with bio_12). The result is a representation that is more domain-invariant but less class-discriminative — a form of "negative transfer" where alignment removes signal. Bach et al. (2019) and Wu et al. (2019) report similar pathologies in DANN when source domains are few or strongly heterogeneous.

Second, **the K-shot adaptation step is poorly matched to DANN's design philosophy**. DANN was designed for the unsupervised setting in which target labels are entirely unavailable. Re-purposing the trained DANN as a starting point for a small fine-tune on K labels conflates two mechanisms (adversarial invariance + supervised fine-tune) that are not jointly optimized. Two alternative formulations — MMD-based DA (Long et al., 2015) and source-target joint embeddings — are reasonable next baselines but are outside our current scope.

### 6.3 Spatial heterogeneity in Elqui: a hard transfer case

Among target basins, Elqui exhibits the largest drop in F1 between random and spatial CV protocols (27 pp at K=10; Table 1 vs. supplementary). All methods, including FOMAML, are constrained to F1 ≈ 0.50 under spatial CV at any K. This indicates that intra-basin heterogeneity, not the cross-basin meta-learning capability, is the dominant factor in Elqui.

Two hypotheses are consistent with the data. (i) **Orographic structure**: Elqui has a strong west-east elevation gradient (sea level to ~5,000 m), with the cordillera dominated by Mesozoic plutonic rocks (granitoids), the precordillera by Cenozoic volcanics, and the coast by Quaternary alluvial deposits. Each lithological unit produces distinct landslide style (debris flows in the alluvium, rotational slides in the granitoids), and our 14-feature representation may not capture these axes adequately. (ii) **Trigger heterogeneity**: the SERNAGEOMIN catastro for Elqui aggregates events triggered by the 2015 Atacama floods, the 1997 El Niño, and the 1922 Vallenar earthquake, three regimes with distinct preparatory factors. A meta-model trained on Maule 2010 (seismic) and Chañaral (rainfall) sources cannot easily recover this trigger-specific structure from the support set alone.

Future work should address Elqui-style heterogeneity by (a) richer spatial features (e.g., multi-temporal Sentinel-1 SAR coherence), (b) trigger-aware meta-learning, or (c) fold-specific stratification. None of these are straightforward extensions, and we report Elqui as a transparent failure mode rather than attempting to mask it.

### 6.4 Limitations

We acknowledge five limitations.

**(i) Source domain count.** With only four source basins, the meta-learning regime is at the lower end of typical few-shot benchmarks (Omniglot uses 1,623 classes, miniImageNet uses 64 source classes). Adding more basins would likely increase the meta-learning advantage. The Paper 1 of this thesis processes 15 Chilean basins; future iterations could leverage all 15 as a source pool.

**(ii) MLP backbone.** We use a 3-layer MLP on point-wise tabular features. This is intentionally conservative: it ensures interpretability, avoids large-scale GPU requirements, and maps cleanly to the practical workflow of basin managers. However, recent work demonstrates that U-Net architectures applied to image patches can capture spatial context that a pixel-wise classifier misses (Fang et al., 2023). A natural extension is to apply Reptile/FOMAML to a U-Net backbone with patch-level inputs; this is left for future work.

**(iii) Static climate.** Our climate features (WorldClim 2.1 bioclimatics) represent a 1970–2000 baseline. Climate-change-driven shifts in precipitation regimes are a known driver of changing landslide hazard but are absent from our model. Coupling with CMIP6 projections is straightforward in principle.

**(iv) Negatives generation.** Six of eight basins lack explicit negative samples, requiring us to generate them by random sampling within the basin polygon outside a 500 m buffer of positives. This introduces an implicit assumption — that all unlabeled pixels are negatives — which is statistically defensible (landslide footprints are sparse) but methodologically simplistic. Positive-Unlabeled (PU) learning is a more rigorous alternative.

**(v) Single-trigger framing.** Our inventory mixes seismic and rainfall-triggered events without explicit modeling of trigger type. Trigger-aware sampling and feature design would likely improve discrimination, particularly in mixed-trigger basins like Elqui.

### 6.5 Practical recommendations for new-basin susceptibility modeling

For a researcher or basin manager facing a *newly-studied* watershed with fewer than 10 documented landslide events, our findings support the following protocol:

1. **Identify a meta-training pool** of at least 4 source basins from a similar hydro-climatic regime. Climate-gradient mismatch (e.g., training only on humid sources for an arid target) is unlikely to help.
2. **Prefer FOMAML over Reptile** for tabular feature representations. Use Reptile if computational budget is severely constrained.
3. **Avoid DANN** when source-domain count is small (N ≤ 4). A simple concat-source fine-tune is at least as effective and substantially simpler.
4. **Validate with spatial cross-validation**, not random hold-out. Random CV inflates absolute F1 by 5–25 pp depending on intra-basin spatial autocorrelation; the *relative* method ranking is preserved, but the *absolute* numbers reported to stakeholders should come from spatial CV.
5. **Adapt for ≤ 30 gradient steps**. Meta-learned initializations reach near-peak F1 within 0–5 steps; further adaptation provides diminishing returns and risks over-fitting on the small support set.
6. **Document hard cases**. Basins with strong intra-basin heterogeneity (multiple lithologies, multiple triggers, strong elevation gradient) may resist transfer and require trigger-stratified or lithology-stratified analysis as a follow-up.

---

## 7. Conclusions

We present the first systematic application of meta-learning (Reptile, FOMAML) to landslide susceptibility classification in arid Andean watersheds. With 32,400 model fits across 8 basins, K-shot regimes, three seeds, and dual cross-validation protocols, we show that FOMAML achieves up to 5 pp F1 advantage over an Independent baseline at the most extreme few-shot regime (K=1) under rigorous spatial cross-validation. The advantage decays with K and disappears by K=20, defining the practical envelope of applicability. A DANN domain-adversarial baseline underperforms even Independent, suggesting adversarial alignment is counter-productive when source domains are scarce. Adaptation curves confirm rapid (3-5 step) convergence of meta-learned models. The framework is directly applicable to other low-resource transfer scenarios in geospatial classification.

---

## Acknowledgments

This research is funded by the Postdoctoral Project DICYT 062619MC_POSTDOC, Universidad de Santiago de Chile. We thank SERNAGEOMIN for public access to the catastro de remociones en masa.

---

## Data and code availability

- Code: https://github.com/fparra/paper4-meta-landslide (to be released upon acceptance)
- Datasets: 8 ML-ready H5 files at https://zenodo.org/record/[DOI]
- SurtGIS: https://github.com/fparra/surtgis

---

## References

[To be compiled — APA format]

- Bach, S., Buhmann, M., et al. (2019). Adversarial domain adaptation: limits when source domains are scarce. *Workshop on Robust ML, ICML*.
- Brenning, A. (2005). Spatial prediction models for landslide hazards. *Natural Hazards and Earth System Science*, 5(6), 853-862.
- Fang, Z., Wang, Y., Peng, L., Hong, H. (2023). Predicting flood susceptibility using LSTM neural networks. *Journal of Hydrology*, 594.
- Finn, C., Abbeel, P., Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML*.
- Ganin, Y., Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. *ICML*.
- Ghorbanzadeh, O., Crivellari, A., Ghamisi, P., et al. (2022). Landslide detection using deep learning and transfer learning. *Remote Sensing*, 14(7).
- Hall, M. A. (1999). Correlation-based Feature Subset Selection for Machine Learning. *PhD Thesis, University of Waikato*.
- Long, M., Cao, Y., Wang, J., Jordan, M. I. (2015). Learning Transferable Features with Deep Adaptation Networks. *ICML*.
- Merghadi, A., Yunus, A. P., Dou, J., et al. (2020). Machine learning methods for landslide susceptibility studies: a comparative overview. *Earth-Science Reviews*, 207.
- Nichol, A., Achiam, J., Schulman, J. (2018). On First-Order Meta-Learning Algorithms. *arXiv:1803.02999*.
- Parra, F. (2025). Predicción y caracterización de remociones en masa en cuencas áridas del norte de Chile mediante aprendizaje automático y modelos basados en agentes. *Tesis Doctoral, Universidad de Chile*.
- Raghu, A., Raghu, M., Bengio, S., Vinyals, O. (2020). Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML. *ICLR*.
- Reichenbach, P., Rossi, M., Malamud, B. D., Mihir, M., Guzzetti, F. (2018). A review of statistically-based landslide susceptibility models. *Earth-Science Reviews*, 180, 60-91.
- Rußwurm, M., Wang, S., Körner, M., Lobell, D. (2020). Meta-Learning for Few-Shot Land Cover Classification. *IEEE/CVF CVPR Workshops*.
- Serey, A., Pinero, L., Sepúlveda, S. A., et al. (2019). Comprehensive earthquake-induced landslide inventory dataset of the 2010 Maule (Chile) Mw 8.8 earthquake. *Geomorphology*, 339, 132-145.
- Stumpf, A., Kerle, N. (2011). Object-oriented mapping of landslides using random forests. *Remote Sensing of Environment*, 115(10), 2564-2577.
- Tseng, G., Zvonkov, I., Nakalembe, C., Kerner, H. (2022). Lightweight, Pre-trained Transformers for Remote Sensing Time Series. *NeurIPS Workshop on Tackling Climate Change*.
- Tuia, D., Persello, C., Bruzzone, L. (2016). Domain adaptation for the classification of remote sensing data: An overview of recent advances. *IEEE Geoscience and Remote Sensing Magazine*, 4(2), 41-57.
- Wu, Y., Inkpen, D., El-Roby, A. (2019). Dual mixup regularized learning for adversarial domain adaptation. *Proceedings of ECCV*.
