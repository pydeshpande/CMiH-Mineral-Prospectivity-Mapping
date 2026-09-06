"""
models.py
=====================================================================
Data-driven and knowledge-driven prospectivity modelling with rigorous
**spatial cross-validation**.

The single most common way MPM studies fool themselves is a random train/test
split: because evidence layers are spatially autocorrelated, a test cell sitting
next to a training cell leaks information and inflates AUC toward ~1.  We avoid
this by partitioning the area into spatial blocks and holding out whole blocks
(GroupKFold).  We then stitch the held-out-block predictions into a *spatially
out-of-fold* prospectivity map -- every cell scored by a model that never saw
its block -- and validate that map against the known occurrences.  That is the
honest number to report to the CMiH-2026 judges; the delivered client map is
refit on all labelled data.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy import ndimage
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
#  labelling
# --------------------------------------------------------------------------- #
def make_labels(shape, occ_rc, neg_ratio=4, pos_buffer=1, neg_exclude=4,
                seed=0):
    """
    Build a labelled training set from known occurrence points.

    positives : each occurrence cell + neighbours within ``pos_buffer`` cells
                (accounts for location uncertainty, boosts the minority class)
    negatives : random cells at least ``neg_exclude`` cells from any occurrence
                (avoids labelling likely-prospective ground as barren), sampled
                at ``neg_ratio`` x the positive count.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    occ = np.zeros(shape, bool)
    occ[occ_rc[:, 0], occ_rc[:, 1]] = True

    pos = ndimage.binary_dilation(occ, iterations=int(pos_buffer))
    pos_rc = np.column_stack(np.where(pos))

    dist_occ = ndimage.distance_transform_edt(~occ)
    allowed = dist_occ >= neg_exclude
    idx = np.column_stack(np.where(allowed))
    n_neg = int(neg_ratio * len(pos_rc))
    sel = rng.choice(len(idx), size=min(n_neg, len(idx)), replace=False)
    neg_rc = idx[sel]

    rc = np.vstack([pos_rc, neg_rc])
    y = np.concatenate([np.ones(len(pos_rc)), np.zeros(len(neg_rc))]).astype(int)
    return rc, y


def block_ids(shape, rc, n_side=5):
    """Assign each (row,col) a spatial-block id in a n_side x n_side grid."""
    H, W = shape
    br = np.clip((rc[:, 0] * n_side // H), 0, n_side - 1)
    bc = np.clip((rc[:, 1] * n_side // W), 0, n_side - 1)
    return br * n_side + bc


def block_of_every_cell(shape, n_side=5):
    H, W = shape
    rr, cc = np.mgrid[0:H, 0:W]
    br = np.clip(rr * n_side // H, 0, n_side - 1)
    bc = np.clip(cc * n_side // W, 0, n_side - 1)
    return br * n_side + bc


# --------------------------------------------------------------------------- #
#  estimators
# --------------------------------------------------------------------------- #
def estimator_factory(kind, seed=0):
    if kind == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=4, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if kind == "LogisticReg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"))
    if kind == "HistGBoost":
        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=400,
            l2_regularization=1.0, random_state=seed)
    raise ValueError(kind)


@dataclass
class MPMResult:
    delivered_map: np.ndarray        # final model, trained on all labels (H,W)
    oof_map: np.ndarray              # spatially out-of-fold map (H,W) for validation
    uncertainty: np.ndarray          # RF per-tree std (H,W)
    cv_auc: dict                     # kind -> spatial-CV AUC (labelled samples)
    best_kind: str
    importances: list                # [(feature, impurity, permutation)]
    feature_names: list


# --------------------------------------------------------------------------- #
#  main driver
# --------------------------------------------------------------------------- #
BASE_KINDS = ("RandomForest", "LogisticReg", "HistGBoost")


def _predict_prob(kind, Xtr, ytr, Xpred, seed=0):
    """Fit `kind` and return P(deposit) for Xpred. 'Ensemble' = mean of bases."""
    if kind == "Ensemble":
        ps = [_predict_prob(k, Xtr, ytr, Xpred, seed) for k in BASE_KINDS]
        return np.mean(ps, axis=0)
    est = estimator_factory(kind, seed)
    est.fit(Xtr, ytr)
    return est.predict_proba(Xpred)[:, 1]


def run_mpm(fb, occ_rc, n_side=5, neg_ratio=4, seed=0,
            kinds=BASE_KINDS + ("Ensemble",)):
    """
    fb : a *built* FeatureBuilder (features.py)
    Returns MPMResult.
    """
    shape = fb.shape
    Xg = fb.grid_X()
    rc, y = make_labels(shape, occ_rc, neg_ratio=neg_ratio, seed=seed)
    X = fb.at(rc)
    groups = block_ids(shape, rc, n_side)
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    folds = [(tr, te) for tr, te in gkf.split(X, y, groups)
             if 0 < y[tr].sum() < len(tr)]

    # ---- spatial-CV AUC on labelled samples, per model ----------------- #
    cv_auc = {}
    for kind in kinds:
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            oof[te] = _predict_prob(kind, X[tr], y[tr], X[te], seed)
        m = ~np.isnan(oof)
        cv_auc[kind] = (float(roc_auc_score(y[m], oof[m]))
                        if len(set(y[m])) == 2 else float("nan"))
    best_kind = max(cv_auc, key=lambda k: cv_auc[k] if cv_auc[k] == cv_auc[k]
                    else -1)

    # ---- spatially out-of-fold FULL-GRID map + fold-ensemble spread ---- #
    cell_block = block_of_every_cell(shape, n_side).ravel()
    oof_map = np.full(shape[0] * shape[1], np.nan)
    fold_preds = []
    for tr, te in folds:
        held = set(np.unique(groups[te]))
        p_full = _predict_prob(best_kind, X[tr], y[tr], Xg, seed)
        fold_preds.append(p_full)
        mask = np.isin(cell_block, list(held))
        oof_map[mask] = p_full[mask]
    oof_map = np.nan_to_num(oof_map, nan=np.nanmin(oof_map)).reshape(shape)
    uncertainty = np.std(np.stack(fold_preds), axis=0).reshape(shape)

    # ---- delivered map (refit on ALL labels) -------------------------- #
    delivered = _predict_prob(best_kind, X, y, Xg, seed).reshape(shape)

    # ---- feature importance (from a fresh RF for stable ranking) ------- #
    rf = estimator_factory("RandomForest", seed).fit(X, y)
    imp = rf.feature_importances_
    perm = permutation_importance(rf, X, y, n_repeats=8, random_state=seed,
                                  n_jobs=-1).importances_mean
    order = np.argsort(imp)[::-1]
    importances = [(fb.names[i], float(imp[i]), float(perm[i])) for i in order]

    return MPMResult(delivered, oof_map, uncertainty, cv_auc, best_kind,
                     importances, list(fb.names))


# --------------------------------------------------------------------------- #
#  knowledge-driven fuzzy-logic overlay (Bonham-Carter fuzzy gamma)
# --------------------------------------------------------------------------- #
def _fuzz_large(a, p=(5, 95)):
    """Increasing membership: larger value -> more favourable (0..1)."""
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def _fuzz_near(dist_m, near=1500.0, far=6000.0):
    """Proximity membership: nearer -> more favourable."""
    return np.clip((far - dist_m) / (far - near + 1e-9), 0, 1)


def fuzzy_prospectivity(layers, commodity="Li", d0_halo_m=1800.0, gamma=0.75):
    """
    Expert/knowledge-driven map combining pegmatite indicator layers with the
    fuzzy-gamma operator.  Independent of the ML model -> a useful benchmark
    and a cross-check on the data-driven result.
    """
    L = layers
    mem = []

    if {"radiometric_eTh", "radiometric_eU", "radiometric_K"} <= L.keys():
        rad = (_fuzz_large(L["radiometric_eTh"]) +
               _fuzz_large(L["radiometric_eU"]) +
               _fuzz_large(L["radiometric_K"])) / 3.0
        mem.append(rad)
    if "magnetics_TMI" in L:
        mem.append(_fuzz_large(-(L["magnetics_TMI"] -
                                 ndimage.gaussian_filter(L["magnetics_TMI"], 25))))
    if "geochem_pathfinder" in L:
        mem.append(_fuzz_large(L["geochem_pathfinder"]))
    if "dist_structure_m" in L:
        mem.append(_fuzz_near(L["dist_structure_m"]))
    if "dist_fertile_granite_m" in L:                 # fractionation-halo band
        d = L["dist_fertile_granite_m"]
        mem.append(np.exp(-0.5 * ((d - d0_halo_m) / 1200.0) ** 2))
    if "lithology" in L:
        lith = np.rint(L["lithology"]).astype(int)
        mem.append(ndimage.gaussian_filter((lith == 2).astype(float), 1.5))

    mem = np.stack(mem, axis=0)
    fuzzy_sum = 1 - np.prod(1 - mem, axis=0)          # fuzzy algebraic sum
    fuzzy_prod = np.prod(mem, axis=0)                 # fuzzy algebraic product
    out = (fuzzy_sum ** gamma) * (fuzzy_prod ** (1 - gamma))
    return (out - out.min()) / (np.ptp(out) + 1e-12)


if __name__ == "__main__":
    from cmih_mpm.synthetic_province import SyntheticPegmatiteProvince
    from cmih_mpm.features import FeatureBuilder
    p = SyntheticPegmatiteProvince(seed=7).generate()
    fb = FeatureBuilder(cell_m=p.cell_m).build(p.layers)
    res = run_mpm(fb, p.occurrences_rc, seed=0)
    print("spatial-CV AUC:", {k: round(v, 3) for k, v in res.cv_auc.items()})
    print("best model    :", res.best_kind)
    print("top features  :")
    for name, gi, pi in res.importances[:8]:
        print(f"   {name:24s} impurity={gi:.3f}  perm={pi:.4f}")
