"""
app.py -- Streamlit web app for the CMiH-2026 Li/REE pegmatite MPM pipeline.

Run:  streamlit run app.py
Deploy:  push to GitHub -> share.streamlit.io (Community Cloud), main file app.py.
"""
from __future__ import annotations

import io
import json
import os
import tempfile

import numpy as np
import streamlit as st

from cmih_mpm.synthetic_province import SyntheticPegmatiteProvince
from cmih_mpm.features import FeatureBuilder
from cmih_mpm.models import run_mpm, fuzzy_prospectivity
from cmih_mpm import validation as V
from cmih_mpm import viz
from cmih_mpm.adapters import DATASOURCES, PATHFINDERS

st.set_page_config(page_title="CMiH-2026 Pegmatite Prospectivity",
                   page_icon="⛏️", layout="wide")


@st.cache_data(show_spinner=False)
def compute(commodity, n, noise, n_occ, neg_ratio, n_side, seed):
    prov = SyntheticPegmatiteProvince(n=n, commodity=commodity, n_occ=n_occ,
                                      noise=noise, seed=seed).generate()
    fb = FeatureBuilder(cell_m=prov.cell_m).build(prov.layers)
    res = run_mpm(fb, prov.occurrences_rc, n_side=n_side,
                  neg_ratio=neg_ratio, seed=0)
    fz = fuzzy_prospectivity(prov.layers, commodity)
    metrics = {
        "oof": V.summary(res.oof_map, prov.occurrences_rc),
        "delivered": V.summary(res.delivered_map, prov.occurrences_rc),
        "fuzzy": V.summary(fz, prov.occurrences_rc),
    }
    return prov, res, fz, metrics


def fig_bytes(fn, *args, **kwargs):
    """Render a viz.* figure (which saves to a ``path=`` argument) -> PNG bytes.

    Passes ``path`` as a keyword (every viz.fig_* has a ``path`` parameter) so
    positional order never matters, and uses mkstemp + immediate close so no
    file handle is held while matplotlib writes (Windows blocks writing to an
    already-open temp file).
    """
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        fn(*args, path=path, **kwargs)
        with open(path, "rb") as fh:
            data = fh.read()
        if not data:
            raise RuntimeError("figure was not written")
        return data
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def geotiff_bytes(pmap, prov):
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=prov.shape[0], width=prov.shape[1],
                      count=1, dtype="float32", crs=prov.crs,
                      transform=Affine(*prov.affine)) as dst:
            dst.write(pmap.astype("float32"), 1)
        return mem.read()


# --------------------------------------------------------------------------- #
st.title("⛏️  Mineral Prospectivity Mapping from Open Data")
st.caption("Li / REE **pegmatite** prospectivity — Critical Minerals Innovation "
           "Hackathon 2026 (Ministry of Mines / JNARDDC). Data-driven ML with "
           "spatial cross-validation, validated against known occurrences.")

with st.sidebar:
    st.header("Configuration")
    commodity = st.radio("Target pegmatite family", ["Li", "REE"],
                         help="Li = LCT (Li-Cs-Ta) · REE = NYF (Nb-Y-F)")
    st.caption("Pathfinders: " + ", ".join(PATHFINDERS[commodity]))
    n = st.select_slider("Grid size (cells/side)", [150, 200, 250, 300, 350], 300)
    noise = st.slider("Observational noise", 0.3, 2.0, 1.0, 0.1)
    n_occ = st.slider("Known occurrences", 20, 120, 55, 5)
    neg_ratio = st.slider("Non-deposit : deposit ratio", 2, 10, 4)
    n_side = st.slider("Spatial-CV blocks (per side)", 3, 8, 5,
                       help="Whole blocks are held out -> honest, leakage-free validation")
    seed = st.number_input("Random seed", 0, 9999, 7)
    st.divider()
    st.markdown("**Open-data mode** — swap the synthetic province for real "
                "GSI-NGDR / Bhukosh / Copernicus rasters via `cmih_mpm.adapters` "
                "(same pipeline). Portals in the *Open data* tab.")

with st.spinner("Running mineral-system model + spatial cross-validation…"):
    prov, res, fz, metrics = compute(commodity, n, noise, n_occ, neg_ratio,
                                     n_side, int(seed))

m = metrics["oof"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Best model", res.best_kind)
c2.metric("Spatial-CV AUC", f"{res.cv_auc[res.best_kind]:.3f}")
c3.metric("Nd (vs random=1)", f"{m['Nd']:.2f}")
c4.metric("Capture @10% area", f"{m['capture_at_10pct']*100:.0f}%")
c5.metric("80% capture in", f"{m['area_for_80pct_capture']*100:.0f}% area")

tabs = st.tabs(["🗺️ Prospectivity", "🧪 Evidence layers", "✅ Validation",
                "📊 Importance", "❓ Uncertainty", "🔬 Truth recovery", "🌐 Open data"])

with tabs[0]:
    st.image(fig_bytes(viz.fig_prospectivity, prov, res.delivered_map,
                       title=f"{commodity} pegmatite prospectivity — delivered "
                             f"({res.best_kind})"),
             use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download GeoTIFF (prospectivity.tif)",
                       geotiff_bytes(res.delivered_map, prov),
                       "prospectivity.tif", "image/tiff")
    d2.download_button("⬇️ Download metrics.json",
                       json.dumps(metrics, indent=2, default=float),
                       "metrics.json", "application/json")

with tabs[1]:
    st.image(fig_bytes(viz.fig_evidence, prov), use_container_width=True)

with tabs[2]:
    st.image(fig_bytes(viz.fig_validation, prov, res.oof_map, fz),
             use_container_width=True)
    st.info("Headline numbers use the **spatially out-of-fold** map (every cell "
            "scored by a model that never saw its block) — the honest estimate "
            "of how the map generalises to *undiscovered* ground.")

with tabs[3]:
    st.image(fig_bytes(viz.fig_importance, res), use_container_width=True)
    st.caption("Permutation importance of engineered evidence layers — the "
               "model's rationale should match pegmatite mineral-system controls "
               "(host lithology, radiometrics, structural & granite proximity).")

with tabs[4]:
    st.image(fig_bytes(viz.fig_uncertainty, prov, res.uncertainty),
             use_container_width=True)

with tabs[5]:
    st.image(fig_bytes(viz.fig_truth_vs_pred, prov, res.delivered_map),
             use_container_width=True)
    st.caption("Synthetic-demo only: the model sees just the evidence layers, "
               "yet recovers the hidden 'true' favourability — evidence the "
               "methodology is sound before applying it to real open data.")

with tabs[6]:
    st.subheader("Open-data sources (India-first)")
    for k, v in DATASOURCES.items():
        st.markdown(f"- **{k}** — {v}")
    st.markdown("""
**Canonical layer ← open dataset**

| layer | open dataset | pegmatite rationale |
|---|---|---|
| lithology | GSI 1:50k geology | fertile granite + schist/gneiss host |
| radiometric K/eTh/eU | airborne gamma-ray | pegmatites are K+Th+U **high** |
| magnetics (TMI) | NGPM/AGRG aeromagnetic | pegmatite/greisen = magnetic **low** |
| gravity | NGPM Bouguer | density contrast |
| geochem | NGCM stream sediment | Li-Cs-Rb-Ta-Sn / LREE-Y-Nb-F anomaly |
| DEM | CartoDEM / Copernicus | resistant quartzo-feldspathic ridges |
| dist. to granite / structure | derived | fractionation halo + fluid pathways |

Point `cmih_mpm.adapters.RealDataLoader` at downloaded GeoTIFFs/vectors — the
features, model and validation run unchanged.
""")
