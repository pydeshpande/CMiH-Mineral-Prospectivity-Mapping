"""
adapters.py
=====================================================================
Load REAL open geoscience data into the same canonical layer dict the pipeline
consumes, so ``features.py`` / ``models.py`` / ``validation.py`` run unchanged
on downloaded data.  Nothing here is downloaded automatically -- you fetch the
open datasets once (portals below), drop them in a folder, and point the loader
at them.  Rasters are reprojected/resampled onto one common analysis grid;
vector occurrences and structures are rasterised; proximities and densities are
derived exactly as in the synthetic province.

------------------------------------------------------------------------------
OPEN DATA SOURCES FOR Li / REE PEGMATITE PROSPECTIVITY (India-first)
------------------------------------------------------------------------------
INDIA (primary — this is a Ministry of Mines / GSI hackathon):
  * GSI NGDR  — National Geoscience Data Repository .......... ngdr.gsi.gov.in
        India's flagship OPEN geoscience portal: geology, geochemistry (NGCM
        stream-sediment incl. Li, Rb, Cs, Nb, Sn, REE), geophysics (NGPM
        gravity + aeromagnetic), lineaments, mineral blocks. START HERE.
  * GSI Bhukosh ............................................. bhukosh.gsi.gov.in
        Map-sheet downloads: 1:50k geology, NGCM geochem points, NGPM Bouguer
        gravity + TMI aeromagnetic grids, mineral occurrences / exploration
        (search "pegmatite", "lithium", "rare metal", "rare earth").
  * ISRO Bhuvan / NRSC ..................................... bhuvan.nrsc.gov.in
        CartoDEM (topography), thematic layers, Landsat/Sentinel mosaics.
  * IBM — Indian Bureau of Mines ............................ ibm.gov.in
        Mineral maps, resource inventories (occurrence validation).

GLOBAL (fill gaps / use where Indian coverage is coarse):
  * Copernicus DEM GLO-30 / SRTM / ASTER GDEM ....... OpenTopography, PlanetaryComputer
  * Sentinel-2 L2A (SWIR alteration ratios: muscovite/greisen, clay, Fe-oxide)
        ................ Copernicus Data Space, MS Planetary Computer (STAC), GEE
  * Landsat 8/9 OLI ............................ USGS EarthExplorer, PlanetaryComputer
  * EMAG2 v3 (global magnetic), WGM2012 / GOCE (gravity) ....... NOAA/NGDC, BGI
  * USGS MRDS / USMIN, OneGeology ........... mrdata.usgs.gov, onegeology.org

------------------------------------------------------------------------------
CANONICAL LAYER  <-  OPEN DATASET  (feature rationale)
------------------------------------------------------------------------------
  lithology                <- GSI 1:50k geology (favourable: fertile granite,
                              metasediment schist/gneiss host)
  radiometric_K/eTh/eU     <- airborne gamma-ray (K%, eTh, eU); pegmatites are
                              K + Th + U HIGH (K-feldspar, mica, monazite/columbite)
  magnetics_TMI            <- NGPM/AGRG RTP aeromagnetic; pegmatite/greisen = LOW
  gravity_resid            <- NGPM Bouguer gravity residual
  geochem_pathfinder       <- NGCM stream sediment. LCT(Li): Li,Cs,Rb,Ta,Sn,Be
                              (+K/Rb, Rb/Sr). NYF(REE): La,Ce,Y,Nb,F,Th
  dem_elevation            <- CartoDEM / Copernicus DEM (resistant peg ridges)
  dist_fertile_granite_m   <- derived: distance to fractionated-granite polygons
  dist_structure_m         <- derived: distance to faults/shear zones/lineaments
  lineament_density        <- derived: kernel density of lineaments
  intersection_density     <- derived: density of fault/lineament intersections
  (optional) sentinel_muscovite / clay / feoxide  <- SWIR band ratios
------------------------------------------------------------------------------
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# pathfinder-element suites by pegmatite family (for building geochem composite)
PATHFINDERS = {
    "Li":  ["Li", "Cs", "Rb", "Ta", "Sn", "Be"],      # LCT family
    "REE": ["La", "Ce", "Nd", "Y", "Nb", "F", "Th"],  # NYF / REE-enriched
}

DATASOURCES = {
    "GSI NGDR":     "https://ngdr.gsi.gov.in",
    "GSI Bhukosh":  "https://bhukosh.gsi.gov.in",
    "ISRO Bhuvan":  "https://bhuvan.nrsc.gov.in",
    "Copernicus DEM/Sentinel": "https://dataspace.copernicus.eu",
    "USGS EarthExplorer":      "https://earthexplorer.usgs.gov",
    "OpenTopography":          "https://opentopography.org",
}


@dataclass
class Grid:
    """Common analysis grid all layers are resampled onto."""
    crs: str
    transform: tuple          # rasterio Affine as 6-tuple (a,b,c,d,e,f)
    shape: tuple              # (rows, cols)
    cell_m: float


# --------------------------------------------------------------------------- #
#  raster / vector helpers (functional for locally-downloaded open data)
# --------------------------------------------------------------------------- #
def make_grid_from_bounds(bounds, cell_m, crs="EPSG:32644"):
    """Build a Grid from (minx,miny,maxx,maxy) in a projected CRS (metres)."""
    from rasterio.transform import from_origin
    minx, miny, maxx, maxy = bounds
    cols = int(np.ceil((maxx - minx) / cell_m))
    rows = int(np.ceil((maxy - miny) / cell_m))
    tr = from_origin(minx, maxy, cell_m, cell_m)
    return Grid(crs=crs, transform=tuple(tr)[:6], shape=(rows, cols),
                cell_m=cell_m)


def align_raster(path, grid: Grid, resampling="bilinear", band=1):
    """Reproject/resample any GeoTIFF onto the common grid -> 2-D float array."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import Affine
    rs = getattr(Resampling, resampling)
    dst = np.full(grid.shape, np.nan, "float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, band), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=Affine(*grid.transform), dst_crs=grid.crs,
            resampling=rs)
    return dst


def rasterize_lonlat_points(lon, lat, grid: Grid, src_crs="EPSG:4326"):
    """Project (lon,lat) occurrence/point coords to grid row,col indices."""
    from rasterio.warp import transform as warp_transform
    from rasterio.transform import Affine, rowcol
    xs, ys = warp_transform(src_crs, grid.crs, list(lon), list(lat))
    tr = Affine(*grid.transform)
    rows, cols = rowcol(tr, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = ((rows >= 0) & (rows < grid.shape[0]) &
          (cols >= 0) & (cols < grid.shape[1]))
    return np.column_stack([rows[ok], cols[ok]]).astype(int)


def rasterize_lines(features, grid: Grid, src_crs="EPSG:4326"):
    """Rasterise line/polygon geometries (GeoJSON-like) to a boolean grid."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from rasterio.warp import transform_geom
    geoms = [transform_geom(src_crs, grid.crs, f) for f in features]
    arr = rasterize(geoms, out_shape=grid.shape, transform=Affine(*grid.transform),
                    fill=0, default_value=1, all_touched=True)
    return arr.astype(bool)


def distance_m(boolean, grid: Grid):
    from scipy import ndimage
    return ndimage.distance_transform_edt(~boolean) * grid.cell_m


def density(boolean, grid: Grid, sigma_cells=8.0):
    from scipy import ndimage
    d = ndimage.gaussian_filter(boolean.astype(float), sigma_cells)
    return (d - d.min()) / (np.ptp(d) + 1e-12)


def geochem_composite(element_rasters: dict, commodity="Li"):
    """Log-standardise pathfinder-element rasters and average -> anomaly index."""
    suite = [e for e in PATHFINDERS[commodity] if e in element_rasters]
    if not suite:
        raise ValueError(f"no pathfinder elements from {PATHFINDERS[commodity]} "
                         f"found in {list(element_rasters)}")
    zs = []
    for e in suite:
        a = np.asarray(element_rasters[e], float)
        a = np.log1p(np.clip(a - np.nanmin(a), 0, None))
        zs.append((a - np.nanmean(a)) / (np.nanstd(a) + 1e-9))
    comp = np.nanmean(zs, axis=0)
    return (comp - np.nanmin(comp)) / (np.ptp(comp[np.isfinite(comp)]) + 1e-12)


# --------------------------------------------------------------------------- #
#  loader
# --------------------------------------------------------------------------- #
class RealDataLoader:
    """
    Assemble the canonical layer dict from locally-downloaded open datasets.

    Example
    -------
    >>> grid = make_grid_from_bounds((600000, 1600000, 660000, 1660000),
    ...                              cell_m=200, crs="EPSG:32644")
    >>> loader = RealDataLoader(grid, commodity="Li")
    >>> loader.add_raster("magnetics_TMI", "ngpm_rtp.tif")
    >>> loader.add_raster("dem_elevation", "cartodem.tif", resampling="bilinear")
    >>> loader.add_radiometrics(k="gr_K.tif", eth="gr_eTh.tif", eu="gr_eU.tif")
    >>> loader.add_geochem_elements({"Li": "Li.tif", "Rb": "Rb.tif", ...})
    >>> loader.add_lithology("geology.tif", favourable_classes=[...])
    >>> loader.add_structures_from_geojson(list_of_line_features)
    >>> loader.add_granite_polygons(list_of_polygon_features)
    >>> layers = loader.finalize()
    >>> occ_rc = loader.occurrences_from_csv("occurrences.csv", "lon", "lat")
    """

    def __init__(self, grid: Grid, commodity="Li"):
        self.grid = grid
        self.commodity = commodity
        self.layers: dict[str, np.ndarray] = {}

    def add_raster(self, name, path, resampling="bilinear", band=1):
        self.layers[name] = align_raster(path, self.grid, resampling, band)
        return self

    def add_radiometrics(self, k, eth, eu):
        self.add_raster("radiometric_K", k)
        self.add_raster("radiometric_eTh", eth)
        self.add_raster("radiometric_eU", eu)
        return self

    def add_geochem_elements(self, element_paths: dict):
        rasters = {e: align_raster(p, self.grid) for e, p in element_paths.items()}
        self.layers["geochem_pathfinder"] = geochem_composite(rasters, self.commodity)
        return self

    def add_lithology(self, path, favourable_classes=None):
        self.add_raster("lithology", path, resampling="nearest")
        self._litho_favourable = favourable_classes
        return self

    def add_structures_from_geojson(self, line_features, src_crs="EPSG:4326"):
        mask = rasterize_lines(line_features, self.grid, src_crs)
        self.layers["dist_structure_m"] = distance_m(mask, self.grid)
        self.layers["lineament_density"] = density(mask, self.grid)
        # intersections: dilated self-overlap of the line network
        from scipy import ndimage
        skel = ndimage.binary_dilation(mask, iterations=1)
        inter = skel & ndimage.binary_dilation(mask, iterations=3) & ~mask
        self.layers["intersection_density"] = density(inter, self.grid, 6.0)
        return self

    def add_granite_polygons(self, polygon_features, src_crs="EPSG:4326"):
        mask = rasterize_lines(polygon_features, self.grid, src_crs)
        self.layers["dist_fertile_granite_m"] = distance_m(mask, self.grid)
        return self

    def occurrences_from_csv(self, path, lon_col="lon", lat_col="lat",
                             src_crs="EPSG:4326"):
        import csv
        lon, lat = [], []
        with open(path) as fh:
            for row in csv.DictReader(fh):
                try:
                    lon.append(float(row[lon_col])); lat.append(float(row[lat_col]))
                except (KeyError, ValueError):
                    continue
        return rasterize_lonlat_points(lon, lat, self.grid, src_crs)

    def finalize(self):
        # impute residual NaNs (edge/no-data) with per-layer nan-median
        for k, v in self.layers.items():
            if np.isnan(v).any():
                med = np.nanmedian(v)
                self.layers[k] = np.where(np.isfinite(v), v, med)
        return self.layers


def print_recipe():
    print(__doc__.split("OPEN DATA SOURCES")[1])


if __name__ == "__main__":
    print("Pathfinder suites:", PATHFINDERS)
    print("\nOpen-data portals:")
    for k, v in DATASOURCES.items():
        print(f"  {k:28s} {v}")
    print("\nRun print_recipe() for the full layer<-dataset mapping.")
