"""Fast sanity tests for the CMiH-2026 MPM pipeline (run: pytest -q)."""
import numpy as np
import pytest

from cmih_mpm.synthetic_province import SyntheticPegmatiteProvince
from cmih_mpm.features import FeatureBuilder
from cmih_mpm.models import run_mpm, fuzzy_prospectivity, make_labels, block_ids
from cmih_mpm import validation as V
from cmih_mpm import adapters as A


@pytest.fixture(scope="module")
def province():
    return SyntheticPegmatiteProvince(n=120, commodity="Li", n_occ=45,
                                      seed=3).generate()


@pytest.fixture(scope="module")
def built(province):
    fb = FeatureBuilder(cell_m=province.cell_m).build(province.layers)
    return province, fb


def test_province(province):
    assert province.shape == (120, 120)
    assert len(province.occurrences_rc) > 20
    assert 0.0 <= province.latent.min() and province.latent.max() <= 1.0
    for a in province.layers.values():
        assert np.isfinite(a).all()


def test_features(built):
    _, fb = built
    assert fb.cube.shape[:2] == (120, 120)
    assert len(fb.names) >= 12
    assert np.isfinite(fb.cube).all()
    assert {"ratio_Th_K", "mag_low", "dem_roughness"} <= set(fb.names)


def test_labels_balanced(province):
    rc, y = make_labels(province.shape, province.occurrences_rc, neg_ratio=4)
    assert y.sum() > 0 and (y == 0).sum() > y.sum()          # more negatives
    g = block_ids(province.shape, rc, 5)
    assert len(np.unique(g)) > 1


def test_model_beats_random(built):
    prov, fb = built
    res = run_mpm(fb, prov.occurrences_rc, n_side=4, seed=0)
    assert res.delivered_map.shape == prov.shape
    assert res.oof_map.shape == prov.shape
    assert res.uncertainty.shape == prov.shape
    # honest spatial-OOF discrimination clearly above chance
    s = V.summary(res.oof_map, prov.occurrences_rc)
    assert s["auc_roc"] > 0.6
    assert s["Nd"] > 1.2
    # the top features should be genuine pegmatite mineral-system controls,
    # not nuisance layers -> the model is learning geology, not artefacts
    controls = {"litho_schist", "rad_anomaly", "rad_K", "ratio_Th_K",
                "ratio_U_K", "mag_low", "mag_analytic_signal", "geochem",
                "dist_structure_m", "dist_fertile_granite_m",
                "lineament_density", "intersection_density"}
    top3 = {name for name, _, _ in res.importances[:3]}
    assert len(top3 & controls) >= 2


def test_success_curve_monotone(built):
    prov, fb = built
    res = run_mpm(fb, prov.occurrences_rc, n_side=4, seed=0)
    area, cap = V.capture_vs_area(res.delivered_map, prov.occurrences_rc)
    assert np.all(np.diff(cap) >= -1e-9)                     # non-decreasing
    assert cap[-1] == pytest.approx(1.0, abs=1e-6)           # all captured at 100%
    assert np.trapezoid(cap, area) > 0.5                     # better than random


def test_fuzzy_range(province):
    fz = fuzzy_prospectivity(province.layers, "Li")
    assert fz.shape == province.shape
    assert 0.0 <= fz.min() and fz.max() <= 1.0


def test_adapters(tmp_path):
    assert set(A.PATHFINDERS) == {"Li", "REE"}
    rng = np.random.default_rng(0)
    rasters = {e: rng.random((20, 20)) for e in A.PATHFINDERS["Li"]}
    comp = A.geochem_composite(rasters, "Li")
    assert comp.shape == (20, 20)
    assert 0.0 <= comp.min() and comp.max() <= 1.0
