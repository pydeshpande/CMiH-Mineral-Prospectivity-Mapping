"""
real_data_example.py  --  RUN THE PIPELINE ON REAL OPEN DATA
=====================================================================
This is a FILL-IN-THE-BLANKS template. You do three things:

  1. Set STUDY_AREA (a bounding box) and UTM_EPSG for your region  (STEP 1)
  2. Download open data and drop the files into the ./data folder    (see DOCUMENTATION.docx, Section 8)
  3. Run:  python real_data_example.py

Every input layer is OPTIONAL except the occurrences CSV (needed to validate).
Missing files are skipped automatically -- the model uses whatever you provide.
Start with just a DEM + occurrences, then add more layers to improve the map.

The folder layout this script expects (create the ones you have):

    data/
      geology.tif            integer lithology classes 0..4  (optional)
      dem.tif                elevation, metres               (recommended)
      magnetics_TMI.tif      total magnetic intensity        (optional)
      gravity.tif            Bouguer/residual gravity        (optional)
      radiometric_K.tif      %K            (optional)
      radiometric_eTh.tif    ppm eTh       (optional)
      radiometric_eU.tif     ppm eU        (optional)
      geochem/               folder of per-element rasters   (optional)
        Li.tif  Rb.tif  Cs.tif  Sn.tif  ...   (names must match element symbols)
      faults.geojson         fault / shear LINE features     (optional)
      granites.geojson       fertile-granite POLYGON features(optional)
      occurrences.csv        columns: lon,lat  (WGS84)       REQUIRED to validate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cmih_mpm.adapters import make_grid_from_bounds, RealDataLoader, PATHFINDERS
from cmih_mpm.features import FeatureBuilder
from cmih_mpm.models import run_mpm, fuzzy_prospectivity
from cmih_mpm import validation as V
from cmih_mpm import viz

# =====================================================================
# STEP 1 -- DEFINE YOUR STUDY AREA
# ---------------------------------------------------------------------
# COMMODITY: "Li" (LCT pegmatites) or "REE" (NYF pegmatites)
COMMODITY = "Li"

# STUDY_AREA = (min_lon, min_lat, max_lon, max_lat) in degrees (WGS84).
# Example below ~ Marlagalla-Allapatna area, Mandya district, Karnataka
# (India's confirmed lithium pegmatite occurrence). Replace with your area.
STUDY_AREA_LONLAT = (76.75, 12.55, 77.05, 12.80)

# UTM_EPSG: the projected (metre) coordinate system for your longitude.
#   India longitude band -> UTM zone -> EPSG
#     < 72 E ............ 42N .... 32642
#     72 - 78 E ......... 43N .... 32643   (Rajasthan/Degana, Karnataka/Mandya)
#     78 - 84 E ......... 44N .... 32644   (Chhattisgarh, Telangana)
#     84 - 90 E ......... 45N .... 32645   (Jharkhand, Bihar mica belt)
#     90 - 96 E ......... 46N .... 32646
UTM_EPSG = "EPSG:32643"

# CELL_M: analysis pixel size in metres (200 m is a good default; 100 m finer).
CELL_M = 200.0

# =====================================================================
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def _read_geojson_features(path):
    """Return a list of GeoJSON geometry dicts from a .geojson FeatureCollection."""
    obj = json.loads(Path(path).read_text())
    feats = obj.get("features", []) if isinstance(obj, dict) else []
    return [f["geometry"] for f in feats if f.get("geometry")]


def _lonlat_to_utm_bounds(bounds_lonlat, dst_crs):
    from rasterio.warp import transform_bounds
    return transform_bounds("EPSG:4326", dst_crs, *bounds_lonlat)


def export_geotiff(pmap, grid, path):
    import rasterio
    from rasterio.transform import Affine
    with rasterio.open(path, "w", driver="GTiff", height=grid.shape[0],
                       width=grid.shape[1], count=1, dtype="float32",
                       crs=grid.crs, transform=Affine(*grid.transform)) as dst:
        dst.write(pmap.astype("float32"), 1)
        dst.set_band_description(1, f"{COMMODITY} pegmatite prospectivity")


def main():
    # build the common analysis grid in projected metres
    utm_bounds = _lonlat_to_utm_bounds(STUDY_AREA_LONLAT, UTM_EPSG)
    grid = make_grid_from_bounds(utm_bounds, cell_m=CELL_M, crs=UTM_EPSG)
    print(f"Study grid: {grid.shape[0]}x{grid.shape[1]} cells @ {CELL_M:.0f} m "
          f"in {UTM_EPSG}")

    ld = RealDataLoader(grid, commodity=COMMODITY)
    loaded = []

    # --- attach each file IF it exists (all optional) ------------------ #
    def has(name):
        p = DATA / name
        return p if p.exists() else None

    if has("geology.tif"):
        ld.add_lithology(str(DATA / "geology.tif")); loaded.append("lithology")
    if has("dem.tif"):
        ld.add_raster("dem_elevation", str(DATA / "dem.tif")); loaded.append("dem")
    if has("magnetics_TMI.tif"):
        ld.add_raster("magnetics_TMI", str(DATA / "magnetics_TMI.tif")); loaded.append("magnetics")
    if has("gravity.tif"):
        ld.add_raster("gravity_resid", str(DATA / "gravity.tif")); loaded.append("gravity")
    if all(has(f"radiometric_{b}.tif") for b in ("K", "eTh", "eU")):
        ld.add_radiometrics(str(DATA / "radiometric_K.tif"),
                            str(DATA / "radiometric_eTh.tif"),
                            str(DATA / "radiometric_eU.tif")); loaded.append("radiometrics")

    geo_dir = DATA / "geochem"
    if geo_dir.is_dir():
        elem = {e: str(geo_dir / f"{e}.tif") for e in PATHFINDERS[COMMODITY]
                if (geo_dir / f"{e}.tif").exists()}
        if elem:
            ld.add_geochem_elements(elem); loaded.append(f"geochem{list(elem)}")

    if has("faults.geojson"):
        ld.add_structures_from_geojson(_read_geojson_features(DATA / "faults.geojson"))
        loaded.append("structures")
    if has("granites.geojson"):
        ld.add_granite_polygons(_read_geojson_features(DATA / "granites.geojson"))
        loaded.append("granite-distance")

    if not loaded:
        print("\n[!] No input files found in ./data —\n"
              "    read DOCUMENTATION.docx Section 8, download at least a DEM\n"
              "    (data/dem.tif) and an occurrences file (data/occurrences.csv),\n"
              "    then run this script again.  To see the pipeline work right\n"
              "    now with no downloads, run:  python run_demo.py\n")
        return

    layers = ld.finalize()
    print("Loaded layers:", ", ".join(loaded))

    # occurrences (required to train + validate)
    occ_csv = DATA / "occurrences.csv"
    if not occ_csv.exists():
        print("\n[!] data/occurrences.csv missing — cannot train/validate.\n"
              "    Provide a CSV with columns lon,lat of known Li/REE pegmatite\n"
              "    occurrences (from NGDR / literature).")
        return
    occ_rc = ld.occurrences_from_csv(str(occ_csv), "lon", "lat")
    print(f"Known occurrences inside area: {len(occ_rc)}")
    if len(occ_rc) < 8:
        print("[!] Very few occurrences in-area — metrics will be noisy. "
              "Consider enlarging STUDY_AREA.")

    # --- run the identical pipeline used by the demo ------------------- #
    fb = FeatureBuilder(cell_m=grid.cell_m).build(layers)
    res = run_mpm(fb, occ_rc, seed=0)
    fz = fuzzy_prospectivity(layers, COMMODITY)
    print(f"Best model: {res.best_kind} | spatial-CV AUC "
          f"{ {k: round(v,3) for k,v in res.cv_auc.items()} }")

    # --- outputs ------------------------------------------------------- #
    viz.fig_prospectivity(prov=_GridProxy(grid, occ_rc), pmap=res.delivered_map,
                          path=str(OUT / "real_prospectivity_map.png"),
                          title=f"{COMMODITY} pegmatite prospectivity (real data)")
    viz.fig_validation(_GridProxy(grid, occ_rc), res.oof_map, fz,
                       str(OUT / "real_validation.png"))
    export_geotiff(res.delivered_map, grid, OUT / "real_prospectivity.tif")

    metrics = {"commodity": COMMODITY, "layers": loaded,
               "best_model": res.best_kind,
               "spatial_cv_auc": {k: round(v, 4) for k, v in res.cv_auc.items()},
               "honest_oof": V.summary(res.oof_map, occ_rc),
               "delivered": V.summary(res.delivered_map, occ_rc)}
    (OUT / "real_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    s = metrics["honest_oof"]
    print("\n-- validation vs known occurrences (honest spatial-OOF) --")
    print(f"  ROC-AUC {s['auc_roc']:.3f} | Nd {s['Nd']:.2f} | "
          f"capture@10% {s['capture_at_10pct']*100:.0f}% | "
          f"capture@20% {s['capture_at_20pct']*100:.0f}%")
    print(f"\nWrote outputs to {OUT}/  (real_prospectivity.tif opens in QGIS)")


class _GridProxy:
    """Minimal object exposing the attributes viz.* reads (shape, occurrences_rc,
    layers, commodity) so the same figure code works for real-data grids."""
    def __init__(self, grid, occ_rc):
        self.shape = grid.shape
        self.occurrences_rc = np.asarray(occ_rc)
        self.commodity = COMMODITY
        self.cell_m = grid.cell_m
        self.affine = grid.transform
        self.crs = grid.crs
        self.layers = {}
        self.latent = None
        self.faults_major = []      # real structures aren't drawn as overlays here
        self.faults_minor = []


if __name__ == "__main__":
    main()
