"""
validation.py
=====================================================================
Validate a prospectivity map against **known mineral occurrences** using the
metrics standard in the MPM literature:

  * ROC / AUC                 -- discrimination (use the spatial OOF map).
  * Success-rate curve        -- cumulative % occurrences captured vs cumulative
                                 % of area explored (cells ranked by score).
                                 Random baseline = the diagonal.  The headline
                                 exploration statistic: "top X% of the area
                                 captures Y% of known deposits".
  * Prediction-Area (P-A)     -- Yousefi & Carranza (2015).  Prediction-rate and
                                 occupied-area curves vs threshold; their
                                 intersection gives the Normalized Density
                                 Nd = prediction_rate / occupied_area
                                 (Nd = 1 random, > 1 favourable).

All functions take a 2-D prospectivity map (higher = more prospective) and the
(N,2) row/col occurrence coordinates.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def _norm(a):
    a = np.asarray(a, float)
    return (a - a.min()) / (np.ptp(a) + 1e-12)


def capture_vs_area(pmap, occ_rc):
    """Exact cumulative-capture curve. Returns area_frac, capture_frac in [0,1]."""
    scores = pmap.ravel()
    N = scores.size
    order = np.argsort(scores)[::-1]
    rank = np.empty(N, int)
    rank[order] = np.arange(N)                       # 0 = most prospective cell
    fi = occ_rc[:, 0] * pmap.shape[1] + occ_rc[:, 1]
    occ_area = np.sort((rank[fi] + 1) / N)           # area fraction at which each occ is captured
    area = np.linspace(1.0 / N, 1.0, 400)
    capture = np.searchsorted(occ_area, area, side="right") / len(occ_area)
    return area, capture


def roc(pmap, occ_rc, n_bg=20000, seed=1):
    """ROC using occurrences as positives and random background as negatives."""
    rng = np.random.default_rng(seed)
    H, W = pmap.shape
    occ = np.zeros((H, W), bool)
    occ[occ_rc[:, 0], occ_rc[:, 1]] = True
    bg_flat = rng.choice(H * W, size=min(n_bg, H * W), replace=False)
    bg = np.zeros(H * W, bool); bg[bg_flat] = True
    bg &= ~occ.ravel()
    pos = pmap[occ]
    neg = pmap.ravel()[bg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(roc_auc_score(y, s))


def prediction_area(pmap, occ_rc, n=250):
    """P-A curves + Normalized Density at the intersection."""
    p = _norm(pmap)
    t = np.linspace(0, 1, n)
    occ_scores = p[occ_rc[:, 0], occ_rc[:, 1]]
    pred_rate = np.array([(occ_scores >= ti).mean() for ti in t]) * 100      # %
    occ_area = np.array([(p >= ti).mean() for ti in t]) * 100                # %
    # standard P-A intersection: prediction_rate = 100 - occupied_area
    k = int(np.argmin(np.abs(pred_rate - (100 - occ_area))))
    nd = pred_rate[k] / (occ_area[k] + 1e-9)
    return dict(t=t, pred_rate=pred_rate, occ_area=occ_area,
                t_star=float(t[k]), pred_rate_star=float(pred_rate[k]),
                area_star=float(occ_area[k]), Nd=float(nd))


def summary(pmap, occ_rc):
    area, cap = capture_vs_area(pmap, occ_rc)
    _, _, auc = roc(pmap, occ_rc)
    pa = prediction_area(pmap, occ_rc)
    ausrc = float(np.trapezoid(cap, area))           # 0.5 random .. ->1 perfect

    def cap_at(a):
        return float(np.interp(a, area, cap))

    def area_for(c):
        # smallest area fraction reaching capture c
        idx = np.argmax(cap >= c)
        return float(area[idx]) if cap[-1] >= c else 1.0

    return dict(
        auc_roc=auc,
        ausrc=ausrc,
        gini=2 * ausrc - 1,
        Nd=pa["Nd"],
        capture_at_5pct=cap_at(0.05),
        capture_at_10pct=cap_at(0.10),
        capture_at_20pct=cap_at(0.20),
        area_for_80pct_capture=area_for(0.80),
        n_occurrences=int(len(occ_rc)),
    )


if __name__ == "__main__":
    from cmih_mpm.synthetic_province import SyntheticPegmatiteProvince
    from cmih_mpm.features import FeatureBuilder
    from cmih_mpm.models import run_mpm, fuzzy_prospectivity
    p = SyntheticPegmatiteProvince(seed=7).generate()
    fb = FeatureBuilder(cell_m=p.cell_m).build(p.layers)
    res = run_mpm(fb, p.occurrences_rc, seed=0)
    fz = fuzzy_prospectivity(p.layers, p.commodity)
    for name, m in [("data-driven (OOF, honest)", res.oof_map),
                    ("data-driven (delivered)", res.delivered_map),
                    ("knowledge-driven fuzzy", fz)]:
        s = summary(m, p.occurrences_rc)
        print(f"\n{name}")
        print(f"  AUC={s['auc_roc']:.3f}  AUSRC={s['ausrc']:.3f}  Nd={s['Nd']:.2f}")
        print(f"  capture @10% area = {s['capture_at_10pct']*100:.0f}%  "
              f"@20% = {s['capture_at_20pct']*100:.0f}%")
        print(f"  area to capture 80% = {s['area_for_80pct_capture']*100:.0f}%")
