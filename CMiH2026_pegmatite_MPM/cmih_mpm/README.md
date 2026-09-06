# Mineral Prospectivity Mapping from Open Data — Li / REE Pegmatites

**Critical Minerals Innovation Hackathon 2026 (CMiH-2026)** · Problem Statement 1
· Ministry of Mines / JNARDDC Nagpur · National Critical Mineral Mission.

> Build a **prospectivity heatmap for lithium or REE pegmatites from open data,
> validated against known mineral occurrences.**

This repository is a complete, runnable solution: a mineral-system-driven feature
stack, a data-driven machine-learning model with **honest spatial cross-validation**,
validation against known occurrences using the standard exploration metrics, a
knowledge-driven fuzzy benchmark, uncertainty mapping, and adapters that plug the
whole thing into real Indian open geoscience data (GSI NGDR / Bhukosh, ISRO Bhuvan,
Copernicus).

The pipeline ships with a physically-motivated **synthetic pegmatite province** so
the method can be demonstrated and validated end-to-end with zero downloads — the
model recovers the hidden "true" favourability at spatial correlation **r = 0.88**.
The identical `features → model → validation` code runs on real rasters via
`cmih_mpm/adapters.py`.

---

## Why this approach

Pegmatite exploration is a *mineral-system* problem, so the evidence layers are
chosen from the geology of LCT (Li-Cs-Ta) and NYF (REE) pegmatites rather than
thrown at a model blindly:

| Control | Evidence layer | Open dataset |
|---|---|---|
| **Source** — fertile fractionated granite | distance to fractionated-granite polygons | GSI 1:50k geology |
| **Halo** — regional fractionation zonation | distance-to-granite *band* (learnt, non-monotone) | derived |
| **Pathway** — shear zones / fractures | distance-to-structure, lineament & intersection density | GSI lineaments |
| **Host / trap** — metasediment schist, gneiss | lithology one-hot | GSI geology |
| **Signature** — K-feldspar, mica, monazite | radiometric **K, eTh, eU high**, Th/K, U/K | airborne gamma-ray |
| **Signature** — no magnetite / greisen | magnetic **low** (residual), analytic signal | NGPM aeromagnetic |
| **Signature** — pathfinder dispersion | Li-Cs-Rb-Ta-Sn (or LREE-Y-Nb-F) geochem index | NGCM stream sediment |
| **Expression** — resistant quartzo-feldspathic bodies | DEM roughness, slope | CartoDEM / Copernicus |

Four things make the result trustworthy rather than just a pretty map:

1. **Spatial cross-validation.** Evidence layers are spatially autocorrelated, so a
   random train/test split leaks neighbours and inflates AUC toward ~1. We hold out
   whole spatial **blocks** (`GroupKFold`) and stitch the held-out-block predictions
   into a *spatially out-of-fold* map — every cell scored by a model that never saw
   its block. That is the number reported to judges; the delivered client map is
   refit on all labels.
2. **Class-imbalance aware.** Occurrences are rare; positives use a 1-cell buffer,
   negatives are sampled away from occurrences at a controlled ratio, and models use
   balanced class weights (see also the ScienceDirect 2024 imbalanced-MPM study).
3. **Validation against known occurrences** with the exploration-standard metrics:
   ROC-AUC, the **success-rate curve** (% occurrences captured vs % area explored),
   and the **Prediction-Area** plot with Normalized Density **Nd** (1 = random).
4. **Explainability + uncertainty.** Permutation importance shows the model ranks
   real controls (host lithology, radiometrics, structural/granite proximity) on top;
   an uncertainty map (spatial-fold prediction spread) flags where the map is thin.

An **ensemble** of Random Forest + Logistic Regression + HistGradientBoosting is
evaluated alongside each base model (following the Cobar-Basin ensemble MPM study),
and a **knowledge-driven fuzzy-gamma** map provides an independent expert benchmark.

---

## Results (synthetic Li province, 300×300 @ 200 m, 59 known occurrences)

| Map | ROC-AUC | Nd | capture @10% area | capture @20% area | 80%-capture in |
|---|---|---|---|---|---|
| **Data-driven — spatial OOF (honest)** | **0.76** | **2.36** | 31% | 54% | 45% of area |
| Data-driven — delivered (refit) | 0.81 | 2.65 | 37% | 56% | 34% of area |
| Knowledge-driven fuzzy (benchmark) | 0.63 | 2.80 | 29% | 42% | 87% of area |

Spatial-CV AUC — RF 0.73 · **LR 0.76** · HGB 0.72 · Ensemble 0.76.
Hidden-signal recovery: spatial-corr **r = 0.88**.

The honest, spatially-held-out map reduces the exploration search space
substantially — capturing over half the known occurrences in the top fifth of the
area — and the data-driven model clearly outperforms the expert fuzzy overlay on
discrimination. (Delivered > OOF is the expected optimism of refitting on all data;
we report the OOF figure as the headline for exactly that reason.)

Figures written to `outputs/`:
`01_evidence_layers` · `02_prospectivity_map` · `03_compare_maps` ·
`04_truth_vs_pred` · `05_validation` · `06_importance` · `07_uncertainty`,
plus `prospectivity.tif` (GeoTIFF), `prospectivity.csv`, `metrics.json`,
`feature_importance.csv`.

---

## Run it

```bash
pip install -r requirements.txt

python run_demo.py            # end-to-end: figures + GeoTIFF + metrics
streamlit run app.py          # interactive web app
pytest -q                     # 7 tests, ~50 s
```

`run_demo.main("REE")` switches the target to REE (NYF) pegmatites (different
pathfinder suite; radiometric Th/U-dominant).

---

## Using real open data

Fetch the open datasets once and point the loader at them — the features, model and
validation run **unchanged**:

```python
from cmih_mpm.adapters import make_grid_from_bounds, RealDataLoader
from cmih_mpm.features import FeatureBuilder
from cmih_mpm.models import run_mpm

grid = make_grid_from_bounds((minx, miny, maxx, maxy), cell_m=200, crs="EPSG:32644")
ld = (RealDataLoader(grid, commodity="Li")
      .add_lithology("geology.tif", favourable_classes=[...])
      .add_radiometrics(k="gr_K.tif", eth="gr_eTh.tif", eu="gr_eU.tif")
      .add_raster("magnetics_TMI", "ngpm_rtp.tif")
      .add_raster("gravity_resid", "ngpm_bouguer.tif")
      .add_raster("dem_elevation", "cartodem.tif")
      .add_geochem_elements({"Li": "Li.tif", "Rb": "Rb.tif", "Cs": "Cs.tif",
                             "Sn": "Sn.tif"})
      .add_structures_from_geojson(fault_line_features)
      .add_granite_polygons(granite_polygon_features))
layers = ld.finalize()
occ_rc = ld.occurrences_from_csv("known_occurrences.csv", "lon", "lat")

res = run_mpm(FeatureBuilder(cell_m=grid.cell_m).build(layers), occ_rc)
```

**Open-data portals (India-first):**

- **GSI NGDR** — National Geoscience Data Repository — <https://ngdr.gsi.gov.in> — *start here* (geology, NGCM geochem, NGPM geophysics, lineaments, mineral blocks)
- **GSI Bhukosh** — <https://bhukosh.gsi.gov.in> — map-sheet geology / geochem / gravity / aeromagnetic / mineral occurrences (search "pegmatite", "lithium", "rare metal")
- **ISRO Bhuvan / NRSC** — <https://bhuvan.nrsc.gov.in> — CartoDEM, thematic & satellite layers
- **IBM** — <https://ibm.gov.in> — mineral resource inventories (occurrence validation)
- **Copernicus** — <https://dataspace.copernicus.eu> — DEM GLO-30, Sentinel-2 SWIR (mica/greisen alteration ratios)
- Global fallbacks: USGS EarthExplorer, OpenTopography, EMAG2, USGS MRDS/USMIN, OneGeology.

---

## Repository

```
cmih_mpm/
  synthetic_province.py   physically-motivated Li/REE pegmatite province + occurrences
  features.py             mineral-system feature engineering (radiometric ratios,
                          magnetic-low residual, DEM roughness, litho one-hot, …)
  models.py               spatial-block CV, RF/LR/HGB + ensemble, fuzzy-gamma overlay
  validation.py           ROC, success-rate curve, Prediction-Area / Nd
  viz.py                  publication-quality figures
  adapters.py             real open-data loaders (GSI NGDR/Bhukosh, Copernicus, …)
run_demo.py               end-to-end deliverable generator
app.py                    Streamlit web app
tests/test_pipeline.py    pytest suite
```

## Limitations & extensions

The demo province is synthetic (by design — it gives ground truth to *prove* the
method). On real data, results depend on open-data coverage and occurrence
completeness. Natural extensions, all compatible with this codebase: Positive-
Unlabeled learning for the negative class; Sentinel-2 SWIR alteration ratios as extra
layers; convolutional / graph models for spatial context; SHAP for per-target
explanations; and prospectivity-conditioned area ranking for drill-target
prioritisation.

*All plant/site identities, where real data is later used, should be anonymised.
Model, code and figures are original.*
