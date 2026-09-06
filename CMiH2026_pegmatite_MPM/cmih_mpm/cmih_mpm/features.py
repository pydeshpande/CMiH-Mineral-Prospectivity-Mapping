"""
features.py
=====================================================================
Turn a canonical dict of *raw* geoscience layers into a stack of engineered
**evidence layers** tuned to the Li/REE pegmatite mineral system, then expose
helpers to sample the stack at points (training) or over the whole grid
(prediction).

The function operates only on a dict ``{name: 2-D array}`` so the SAME code
runs on the synthetic province and on real GSI-Bhukosh / Copernicus rasters
loaded by ``adapters.py`` -- provided the raw layer names match the canonical
set below.

Canonical raw layers expected (missing ones are skipped gracefully):
    lithology, radiometric_K, radiometric_eTh, radiometric_eU,
    magnetics_TMI, gravity_resid, geochem_pathfinder, dem_elevation,
    dist_fertile_granite_m, dist_structure_m,
    lineament_density, intersection_density
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _z(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)


def _residual(a, sigma):
    """High-pass residual = layer minus its regional (low-pass) trend."""
    a = np.asarray(a, float)
    return a - ndimage.gaussian_filter(a, sigma, mode="reflect")


def _roughness(dem, size=5):
    mean = ndimage.uniform_filter(dem, size)
    sq = ndimage.uniform_filter(dem * dem, size)
    return np.sqrt(np.clip(sq - mean * mean, 0, None))


def _slope(dem, cell_m):
    gy, gx = np.gradient(dem, cell_m)
    return np.hypot(gx, gy)


# lithology class -> readable name (matches synthetic_province encoding)
LITHO_NAMES = {0: "gneiss", 1: "granite", 2: "schist", 3: "metamafic", 4: "cover"}


class FeatureBuilder:
    """Build the engineered evidence-layer cube."""

    def __init__(self, cell_m=200.0):
        self.cell_m = cell_m
        self.names: list[str] = []
        self.cube: np.ndarray | None = None      # (H, W, F)

    def build(self, layers: dict) -> "FeatureBuilder":
        feats: dict[str, np.ndarray] = {}
        L = layers

        # --- radiometrics: pegmatites are K + eTh + eU HIGH -------------- #
        if "radiometric_K" in L:
            feats["rad_K"] = L["radiometric_K"]
        if {"radiometric_eTh", "radiometric_K"} <= L.keys():
            feats["ratio_Th_K"] = L["radiometric_eTh"] / (L["radiometric_K"] + 0.3)
        if {"radiometric_eU", "radiometric_K"} <= L.keys():
            feats["ratio_U_K"] = L["radiometric_eU"] / (L["radiometric_K"] + 0.3)
        if {"radiometric_eTh", "radiometric_eU"} <= L.keys():
            # composite radiometric fertility anomaly
            feats["rad_anomaly"] = (_z(L["radiometric_eTh"]) +
                                    _z(L["radiometric_eU"]) +
                                    0.5 * _z(L["radiometric_K"]))

        # --- magnetics: pegmatite/greisen = magnetic LOW ---------------- #
        if "magnetics_TMI" in L:
            res = _residual(L["magnetics_TMI"], sigma=25)
            feats["mag_low"] = -res                       # high where mag low
            gy, gx = np.gradient(L["magnetics_TMI"])
            feats["mag_analytic_signal"] = np.hypot(gx, gy)

        # --- gravity ----------------------------------------------------- #
        if "gravity_resid" in L:
            feats["grav_resid"] = _residual(L["gravity_resid"], sigma=30)

        # --- geochemistry ------------------------------------------------ #
        if "geochem_pathfinder" in L:
            feats["geochem"] = L["geochem_pathfinder"]

        # --- topography -------------------------------------------------- #
        if "dem_elevation" in L:
            feats["dem_roughness"] = _roughness(L["dem_elevation"])
            feats["dem_slope"] = _slope(L["dem_elevation"], self.cell_m)

        # --- proximities (already engineered by loader) ------------------ #
        for k in ("dist_fertile_granite_m", "dist_structure_m",
                  "lineament_density", "intersection_density"):
            if k in L:
                feats[k] = L[k]

        # --- lithology one-hot (let the model learn favourable units) ---- #
        if "lithology" in L:
            lith = np.rint(L["lithology"]).astype(int)
            for cls, name in LITHO_NAMES.items():
                if (lith == cls).any():
                    feats[f"litho_{name}"] = ndimage.gaussian_filter(
                        (lith == cls).astype(float), 1.5)

        self.names = list(feats)
        self.cube = np.stack([np.asarray(feats[n], float) for n in self.names],
                             axis=-1)
        return self

    # -- sampling ------------------------------------------------------- #
    def at(self, rc: np.ndarray) -> np.ndarray:
        """Feature rows at (N,2) row,col integer coordinates."""
        rc = np.asarray(rc, int)
        return self.cube[rc[:, 0], rc[:, 1], :]

    def grid_X(self) -> np.ndarray:
        """(H*W, F) design matrix for full-grid prediction."""
        H, W, F = self.cube.shape
        return self.cube.reshape(H * W, F)

    @property
    def shape(self):
        return self.cube.shape[:2]


if __name__ == "__main__":
    from cmih_mpm.synthetic_province import SyntheticPegmatiteProvince
    p = SyntheticPegmatiteProvince(seed=7).generate()
    fb = FeatureBuilder(cell_m=p.cell_m).build(p.layers)
    print("features:", len(fb.names))
    for n in fb.names:
        print("  ", n)
    print("cube", fb.cube.shape, "| grid_X", fb.grid_X().shape)
