"""
viz.py -- publication-quality figures for the CMiH-2026 MPM deliverable.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cmih_mpm import validation as V

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.bbox": "tight",
})

PROSPECT_CMAP = "magma"


def _faults(ax, prov, lw_major=1.6, lw_minor=0.7, color="cyan"):
    for pl in prov.faults_major:
        ax.plot(pl[:, 1], pl[:, 0], color=color, lw=lw_major, alpha=0.8)
    for pl in prov.faults_minor:
        ax.plot(pl[:, 1], pl[:, 0], color=color, lw=lw_minor, alpha=0.5)


def _occ(ax, occ_rc, **kw):
    style = dict(s=14, facecolor="none", edgecolor="lime", linewidths=1.1)
    style.update(kw)
    ax.scatter(occ_rc[:, 1], occ_rc[:, 0], **style)


def fig_evidence(prov, path):
    keys = ["lithology", "radiometric_K", "radiometric_eTh", "radiometric_eU",
            "magnetics_TMI", "gravity_resid", "geochem_pathfinder",
            "dem_elevation", "dist_fertile_granite_m", "dist_structure_m",
            "lineament_density", "intersection_density"]
    cmaps = {"lithology": "tab10", "magnetics_TMI": "RdBu_r",
             "gravity_resid": "RdBu_r", "dem_elevation": "terrain",
             "geochem_pathfinder": "viridis"}
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, k in zip(axes.ravel(), keys):
        im = ax.imshow(prov.layers[k], cmap=cmaps.get(k, "cividis"))
        ax.set_title(f"{k}\n[{prov.units[k]}]", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Open-data evidence layers — {prov.commodity} pegmatite "
                 f"province ({prov.shape[0]}×{prov.shape[1]} @ "
                 f"{prov.cell_m:.0f} m)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path); plt.close(fig)


def fig_prospectivity(prov, pmap, path, title=None):
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(pmap, cmap=PROSPECT_CMAP, vmin=0, vmax=1)
    _faults(ax, prov); _occ(ax, prov.occurrences_rc)
    ax.set_title(title or f"{prov.commodity} pegmatite prospectivity heatmap")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("prospectivity  (0 – 1)")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", mfc="none", mec="lime", ls="", label="known occurrence"),
        Line2D([0], [0], color="cyan", lw=1.6, label="interpreted structure")],
        loc="lower right", fontsize=8, framealpha=0.85)
    fig.savefig(path); plt.close(fig)


def fig_compare(prov, maps: dict, path):
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4))
    if n == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, maps.items()):
        im = ax.imshow(m, cmap=PROSPECT_CMAP, vmin=0, vmax=1)
        _occ(ax, prov.occurrences_rc, s=9)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Prospectivity: honest (spatial OOF) vs delivered vs "
                 "knowledge-driven", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path); plt.close(fig)


def fig_truth_vs_pred(prov, pmap, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    im0 = axes[0].imshow(prov.latent, cmap=PROSPECT_CMAP, vmin=0, vmax=1)
    axes[0].set_title("Ground-truth latent favourability\n(hidden from model)")
    im1 = axes[1].imshow(pmap, cmap=PROSPECT_CMAP, vmin=0, vmax=1)
    axes[1].set_title("Model-predicted prospectivity\n(from evidence layers only)")
    for ax, im in ((axes[0], im0), (axes[1], im1)):
        _occ(ax, prov.occurrences_rc, s=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    r = np.corrcoef(prov.latent.ravel(), pmap.ravel())[0, 1]
    fig.suptitle(f"Recovery of the hidden mineral-system signal "
                 f"(spatial-corr r = {r:.2f})", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path); plt.close(fig)
    return r


def fig_validation(prov, oof_map, fuzzy_map, path):
    occ = prov.occurrences_rc
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # ROC
    f1, t1, a1 = V.roc(oof_map, occ)
    f2, t2, a2 = V.roc(fuzzy_map, occ)
    axes[0].plot(f1, t1, lw=2, label=f"data-driven OOF (AUC {a1:.2f})")
    axes[0].plot(f2, t2, lw=1.6, ls="--", label=f"fuzzy (AUC {a2:.2f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set(xlabel="false-positive rate", ylabel="true-positive rate",
                title="ROC vs known occurrences")
    axes[0].legend(fontsize=8)

    # success-rate curve
    ar, cp = V.capture_vs_area(oof_map, occ)
    ar2, cp2 = V.capture_vs_area(fuzzy_map, occ)
    axes[1].plot(ar * 100, cp * 100, lw=2, label="data-driven OOF")
    axes[1].plot(ar2 * 100, cp2 * 100, lw=1.6, ls="--", label="fuzzy")
    axes[1].plot([0, 100], [0, 100], "k:", lw=1, label="random")
    for a in (10, 20):
        c = np.interp(a / 100, ar, cp) * 100
        axes[1].annotate(f"{c:.0f}%", (a, c), fontsize=8,
                         xytext=(a + 2, c - 8), color="C0")
        axes[1].axvline(a, color="grey", lw=0.5, ls=":")
    axes[1].set(xlabel="% of area explored (ranked)",
                ylabel="% of known occurrences captured",
                title="Success-rate curve")
    axes[1].legend(fontsize=8)

    # Prediction-Area
    pa = V.prediction_area(oof_map, occ)
    axes[2].plot(pa["t"], pa["pred_rate"], lw=2, color="C0",
                 label="prediction rate (% occ)")
    ax2 = axes[2].twinx()
    ax2.plot(pa["t"], pa["occ_area"], lw=2, color="C3",
             label="occupied area (%)")
    ax2.invert_yaxis()
    axes[2].axvline(pa["t_star"], color="grey", ls=":", lw=1)
    axes[2].set(xlabel="prospectivity threshold",
                ylabel="prediction rate (%)",
                title=f"Prediction-Area plot   (Nd = {pa['Nd']:.2f})")
    axes[2].set_ylabel("prediction rate (%)", color="C0")
    ax2.set_ylabel("occupied area (%)", color="C3")
    fig.suptitle("Validation against known Li/REE pegmatite occurrences",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path); plt.close(fig)


def fig_importance(res, path, top=14):
    imp = res.importances[:top][::-1]
    names = [x[0] for x in imp]
    perm = [x[2] for x in imp]
    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.barh(names, perm, color="teal")
    ax.set(xlabel="permutation importance (Δ score)",
           title="Evidence-layer importance (interpretability)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_uncertainty(prov, unc, path):
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(unc, cmap="cividis")
    _occ(ax, prov.occurrences_rc, s=9)
    ax.set_title("Model uncertainty (spatial-fold prediction spread)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="std. dev.")
    fig.savefig(path); plt.close(fig)
