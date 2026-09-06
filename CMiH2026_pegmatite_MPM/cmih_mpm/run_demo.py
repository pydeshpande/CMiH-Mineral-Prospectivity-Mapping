"""
run_demo.py -- end-to-end CMiH-2026 Li/REE pegmatite prospectivity demo.

Generates (in ./outputs):
    01_evidence_layers.png      raw open-data evidence layers
    02_prospectivity_map.png    delivered prospectivity heatmap (client product)
    03_compare_maps.png         honest OOF vs delivered vs knowledge-driven
    04_truth_vs_pred.png        recovery of the hidden mineral-system signal
    05_validation.png           ROC + success-rate + Prediction-Area
    06_importance.png           evidence-layer importance (explainability)
    07_uncertainty.png          spatial-fold prediction spread
    prospectivity.tif           GeoTIFF (georeferenced heatmap) -- the deliverable
    prospectivity.csv           x,y,prospectivity table
    metrics.json                validation metrics vs known occurrences
    feature_importance.csv      ranked importances
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from cmih_mpm.synthetic_province import SyntheticPegmatiteProvince
from cmih_mpm.features import FeatureBuilder
from cmih_mpm.models import run_mpm, fuzzy_prospectivity
from cmih_mpm import validation as V
from cmih_mpm import viz

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def export_geotiff(pmap, prov, path):
    try:
        import rasterio
        from rasterio.transform import Affine
        with rasterio.open(
                path, "w", driver="GTiff", height=prov.shape[0],
                width=prov.shape[1], count=1, dtype="float32",
                crs=prov.crs, transform=Affine(*prov.affine)) as dst:
            dst.write(pmap.astype("float32"), 1)
            dst.set_band_description(1, f"{prov.commodity} pegmatite prospectivity")
        return True
    except Exception as e:
        print("  (GeoTIFF export skipped:", e, ")")
        return False


def export_csv(pmap, prov, path):
    from rasterio.transform import Affine
    tr = Affine(*prov.affine)
    rows, cols = np.mgrid[0:prov.shape[0], 0:prov.shape[1]]
    xs, ys = tr * (cols.ravel() + 0.5, rows.ravel() + 0.5)
    np.savetxt(path, np.column_stack([xs, ys, pmap.ravel()]),
               delimiter=",", header="x,y,prospectivity", comments="",
               fmt=["%.1f", "%.1f", "%.5f"])


def main(commodity="Li", seed=7):
    print(f"== CMiH-2026 MPM demo :: {commodity} pegmatites ==")
    prov = SyntheticPegmatiteProvince(commodity=commodity, seed=seed).generate()
    print(f"province {prov.shape}  |  {len(prov.occurrences_rc)} known occurrences")

    fb = FeatureBuilder(cell_m=prov.cell_m).build(prov.layers)
    print(f"engineered {len(fb.names)} evidence layers")

    res = run_mpm(fb, prov.occurrences_rc, seed=0)
    fz = fuzzy_prospectivity(prov.layers, commodity)
    print(f"best model: {res.best_kind}  |  spatial-CV AUC "
          f"{ {k: round(v,3) for k,v in res.cv_auc.items()} }")

    # figures
    viz.fig_evidence(prov, OUT / "01_evidence_layers.png")
    viz.fig_prospectivity(prov, res.delivered_map, OUT / "02_prospectivity_map.png",
                          title=f"{commodity} pegmatite prospectivity — delivered "
                                f"(model: {res.best_kind})")
    viz.fig_compare(prov, {"data-driven (OOF, honest)": res.oof_map,
                           "data-driven (delivered)": res.delivered_map,
                           "knowledge-driven (fuzzy)": fz},
                    OUT / "03_compare_maps.png")
    r = viz.fig_truth_vs_pred(prov, res.delivered_map, OUT / "04_truth_vs_pred.png")
    viz.fig_validation(prov, res.oof_map, fz, OUT / "05_validation.png")
    viz.fig_importance(res, OUT / "06_importance.png")
    viz.fig_uncertainty(prov, res.uncertainty, OUT / "07_uncertainty.png")

    # georeferenced deliverables
    export_geotiff(res.delivered_map, prov, OUT / "prospectivity.tif")
    export_csv(res.delivered_map, prov, OUT / "prospectivity.csv")

    # validation metrics (headline = honest spatial-OOF map)
    metrics = {
        "commodity": commodity,
        "best_model": res.best_kind,
        "spatial_cv_auc": {k: round(v, 4) for k, v in res.cv_auc.items()},
        "truth_recovery_corr": round(float(r), 4),
        "honest_oof": V.summary(res.oof_map, prov.occurrences_rc),
        "delivered": V.summary(res.delivered_map, prov.occurrences_rc),
        "knowledge_driven_fuzzy": V.summary(fz, prov.occurrences_rc),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    with open(OUT / "feature_importance.csv", "w") as fh:
        fh.write("feature,impurity_importance,permutation_importance\n")
        for name, gi, pi in res.importances:
            fh.write(f"{name},{gi:.5f},{pi:.5f}\n")

    s = metrics["honest_oof"]
    print("\n-- validation vs known occurrences (honest spatial-OOF) --")
    print(f"  ROC-AUC ............... {s['auc_roc']:.3f}")
    print(f"  Normalized density Nd . {s['Nd']:.2f}  (1 = random)")
    print(f"  capture @10% area ..... {s['capture_at_10pct']*100:.0f}%")
    print(f"  capture @20% area ..... {s['capture_at_20pct']*100:.0f}%")
    print(f"  search-space reduction  capture 80% of occ. in "
          f"{s['area_for_80pct_capture']*100:.0f}% of area")
    print(f"\nwrote deliverables to {OUT}/")
    return metrics


if __name__ == "__main__":
    main("Li", seed=7)
