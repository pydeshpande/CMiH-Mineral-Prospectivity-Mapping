"""
synthetic_province.py
=====================================================================
A physically-motivated synthetic geoscience province for demonstrating and
validating the Li / REE **pegmatite** prospectivity pipeline end-to-end,
fully reproducibly, with no external downloads.

Why synthetic?  The CMiH-2026 pipeline is data-source agnostic (see
``adapters.py`` for the real GSI Bhukosh / Copernicus / Sentinel loaders).
To prove the *method* works we need ground truth we do not normally have in
the field: a known latent "true" prospectivity from which occurrences are
drawn.  The generator therefore builds a small greenstone-hosted pegmatite
field the way a real one forms, then thins a point process by the true
mineral-system favourability to place the "known occurrences".  The learner
sees ONLY the noisy geophysical/geochemical/lithological layers and the
occurrence points -- never the latent field -- exactly as in reality.

Encoded pegmatite (LCT / NYF) mineral system controls
-----------------------------------------------------
  SOURCE     : highly fractionated, peraluminous "fertile" granite pluton(s)
  PATHWAY    : crustal-scale shear zones + second-order fractures
  HALO       : a *distance band* out from the parent granite (too close =
               barren/greisen, too far = pegmatites pinch out) -- the
               classic regional zonation (Be -> Li -> REE outward)
  HOST/TRAP  : favourable metasedimentary schist / gneiss, fold + fault traps
  SIGNATURES : radiometric K + eTh + eU HIGH (K-feldspar, mica, monazite,
               columbite), magnetic LOW (no magnetite / magnetite destruction),
               pathfinder geochem anomaly (Li-Cs-Rb-Ta-Sn or LREE-Y-Nb-F),
               resistant quartzo-feldspathic ridges in the DEM.

Everything is returned as plain numpy arrays on a regular grid together with
a valid UTM affine + CRS so it can be written to GeoTIFF identically to real
data.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy import ndimage


# --------------------------------------------------------------------------- #
#  low-level field builders
# --------------------------------------------------------------------------- #
def _grf(shape, corr_len, rng):
    """Smooth Gaussian random field: white noise low-pass filtered, z-scored."""
    w = rng.standard_normal(shape)
    f = ndimage.gaussian_filter(w, sigma=corr_len, mode="reflect")
    return (f - f.mean()) / (f.std() + 1e-9)


def _fractal(shape, rng, octaves=6, persistence=0.55, base=48.0):
    """Fractional-Brownian-like field for topography."""
    out = np.zeros(shape)
    amp = 1.0
    for o in range(octaves):
        out += amp * _grf(shape, base / (2 ** o), rng)
        amp *= persistence
    return (out - out.min()) / (np.ptp(out) + 1e-9)


def _blob(shape, center, radius, softness, rng, jitter=0.35):
    """Irregular pluton body as a soft-edged, noise-perturbed disc (0..1)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r = np.hypot(xx - center[1], yy - center[0])
    edge = radius * (1.0 + jitter * _grf(shape, radius * 0.25, rng))
    return 1.0 / (1.0 + np.exp((r - edge) / max(softness, 1e-3)))


def _polyline_mask(shape, pts, half_width):
    """Rasterise a polyline (list of (row,col)) to a boolean corridor."""
    m = np.zeros(shape, bool)
    pts = np.asarray(pts, float)
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 2
        rr = np.linspace(a[0], b[0], n).round().astype(int)
        cc = np.linspace(a[1], b[1], n).round().astype(int)
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        m[rr[ok], cc[ok]] = True
    if half_width > 0:
        m = ndimage.binary_dilation(m, iterations=int(half_width))
    return m


def _norm(a):
    a = np.asarray(a, float)
    return (a - a.min()) / (np.ptp(a) + 1e-12)


# --------------------------------------------------------------------------- #
#  container
# --------------------------------------------------------------------------- #
@dataclass
class Province:
    layers: dict                 # name -> 2-D float array (the raw evidence)
    units: dict                  # name -> str
    occurrences_rc: np.ndarray   # (N,2) row,col of KNOWN Li/REE pegmatite points
    latent: np.ndarray           # 2-D true favourability (ground truth, hidden)
    faults_major: list           # list of polylines (for plotting)
    faults_minor: list
    cell_m: float
    affine: tuple                # rasterio-style (a,b,c,d,e,f)
    crs: str
    commodity: str
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.latent.shape


# --------------------------------------------------------------------------- #
#  generator
# --------------------------------------------------------------------------- #
class SyntheticPegmatiteProvince:
    """
    Parameters
    ----------
    n            : grid side (cells).  300 -> 60x60 km at 200 m.
    cell_m       : cell size in metres.
    commodity    : 'Li' (LCT) or 'REE' (NYF) -- changes pathfinder suite +
                   which radiometric channels dominate.
    n_occ        : approximate number of known occurrences to plant.
    noise        : global multiplier on observational noise (realism knob).
    seed         : RNG seed.
    """

    def __init__(self, n=300, cell_m=200.0, commodity="Li",
                 n_occ=55, noise=1.0, seed=7):
        assert commodity in ("Li", "REE")
        self.n = int(n)
        self.cell_m = float(cell_m)
        self.commodity = commodity
        self.n_occ = int(n_occ)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)

    # -- public ----------------------------------------------------------- #
    def generate(self) -> Province:
        n, rng = self.n, self.rng
        shape = (n, n)

        litho, fertile, host_schist, meta_mafic = self._lithology(shape)
        faults_major, faults_minor, fault_maj_mask, fault_min_mask = \
            self._structures(shape)

        # distances (in cells) -- also reused as features downstream
        d_gran = ndimage.distance_transform_edt(fertile < 0.5)
        d_fault = ndimage.distance_transform_edt(~(fault_maj_mask | fault_min_mask))

        rad_K, rad_Th, rad_U, peg_zone = self._radiometrics(
            shape, litho, fertile, host_schist, d_gran, d_fault)
        tmi = self._magnetics(shape, litho, fertile, meta_mafic, peg_zone)
        grav = self._gravity(shape, litho, fertile, meta_mafic)
        dem = self._topography(shape, fault_maj_mask, peg_zone)
        geochem = self._geochem(shape, peg_zone, d_fault)

        lin_density = ndimage.gaussian_filter(
            (fault_maj_mask | fault_min_mask).astype(float), 8.0)
        inter_density = self._intersection_density(fault_maj_mask, fault_min_mask)

        # ---- latent TRUE prospectivity (hidden ground truth) ------------ #
        d0 = 9.0        # fractionation-halo sweet-spot distance (cells ~1.8 km)
        halo = np.exp(-0.5 * ((d_gran - d0) / 5.0) ** 2)          # band, not blob
        struct = np.exp(-d_fault / 6.0)
        inter = self._intersection_density(fault_maj_mask, fault_min_mask)
        radio_anom = _norm(rad_Th / (rad_K + 0.3)) * 0.6 + _norm(rad_U) * 0.4
        mag_low = _norm(-tmi)
        geo_anom = _norm(geochem)

        latent = (0.24 * host_schist +
                  0.22 * halo +
                  0.16 * struct +
                  0.10 * _norm(inter) +
                  0.12 * radio_anom +
                  0.08 * mag_low +
                  0.08 * geo_anom)
        latent = _norm(ndimage.gaussian_filter(latent, 1.2))
        latent *= (fertile < 0.85)            # not inside the barren pluton core

        occ_rc = self._plant_occurrences(latent)

        layers = {
            "lithology":      litho.astype(float),
            "radiometric_K":  rad_K,
            "radiometric_eTh": rad_Th,
            "radiometric_eU": rad_U,
            "magnetics_TMI":  tmi,
            "gravity_resid":  grav,
            "geochem_pathfinder": geochem,
            "dem_elevation":  dem,
            "dist_fertile_granite_m": d_gran * self.cell_m,
            "dist_structure_m":      d_fault * self.cell_m,
            "lineament_density":     _norm(lin_density),
            "intersection_density":  _norm(inter_density),
        }
        units = {
            "lithology": "class",
            "radiometric_K": "%K",
            "radiometric_eTh": "ppm eTh",
            "radiometric_eU": "ppm eU",
            "magnetics_TMI": "nT (RTP)",
            "gravity_resid": "mGal",
            "geochem_pathfinder": ("Li-Cs-Rb-Ta-Sn index" if self.commodity == "Li"
                                   else "LREE-Y-Nb-F index"),
            "dem_elevation": "m",
            "dist_fertile_granite_m": "m",
            "dist_structure_m": "m",
            "lineament_density": "0-1",
            "intersection_density": "0-1",
        }

        affine = (self.cell_m, 0.0, 600000.0,
                  0.0, -self.cell_m, 1_650_000.0 + n * self.cell_m)
        return Province(
            layers=layers, units=units, occurrences_rc=occ_rc, latent=latent,
            faults_major=faults_major, faults_minor=faults_minor,
            cell_m=self.cell_m, affine=affine, crs="EPSG:32644",
            commodity=self.commodity,
            meta=dict(n=n, halo_d0_cells=d0, n_occ=len(occ_rc), noise=self.noise),
        )

    # -- components ------------------------------------------------------- #
    def _lithology(self, shape):
        """0 gneiss basement, 1 fertile granite, 2 schist host, 3 metamafic, 4 cover."""
        rng = self.rng
        domain = _grf(shape, 40, rng)
        litho = np.full(shape, 0, int)
        litho[domain > 0.4] = 2                       # metasediment schist belt
        litho[domain < -0.7] = 3                      # metamafic / amphibolite
        cover = _grf(shape, 55, rng)
        litho[cover > 1.1] = 4                         # young cover

        # fertile granite plutons (blobs); the first is the fractionated parent
        fertile = np.zeros(shape)
        for c, r, s in [((0.34, 0.30), 26, 3.2),
                        ((0.70, 0.66), 18, 2.6)]:
            fertile = np.maximum(fertile, _blob(shape, (c[0]*shape[0], c[1]*shape[1]),
                                                r, s, rng))
        litho[fertile > 0.5] = 1
        host_schist = ndimage.gaussian_filter((litho == 2).astype(float), 2.0)
        meta_mafic = (litho == 3).astype(float)
        return litho, fertile, _norm(host_schist), meta_mafic

    def _structures(self, shape):
        rng = self.rng
        n = shape[0]

        def wobble(p0, p1, k=6, amp=0.05):
            t = np.linspace(0, 1, k)
            base = np.outer(1 - t, p0) + np.outer(t, p1)
            base[1:-1, 0] += rng.normal(0, amp * n, k - 2)
            base[1:-1, 1] += rng.normal(0, amp * n, k - 2)
            return base

        major = [wobble((0.05*n, 0.15*n), (0.95*n, 0.55*n)),
                 wobble((0.10*n, 0.85*n), (0.85*n, 0.20*n))]
        minor = []
        for _ in range(9):
            a = (rng.uniform(0.1, 0.9)*n, rng.uniform(0.1, 0.9)*n)
            ang = rng.uniform(0, np.pi)
            L = rng.uniform(0.12, 0.30) * n
            b = (a[0] + L*np.sin(ang), a[1] + L*np.cos(ang))
            minor.append(np.array([a, b]))

        maj_mask = np.zeros(shape, bool)
        for pl in major:
            maj_mask |= _polyline_mask(shape, pl, 1)
        min_mask = np.zeros(shape, bool)
        for pl in minor:
            min_mask |= _polyline_mask(shape, pl, 0)
        return major, minor, maj_mask, min_mask

    def _intersection_density(self, maj, minor):
        # crossings ~ where dilated skeletons overlap
        cross = ndimage.binary_dilation(maj, iterations=2) & \
                ndimage.binary_dilation(minor, iterations=2)
        return ndimage.gaussian_filter(cross.astype(float), 6.0)

    def _radiometrics(self, shape, litho, fertile, host_schist, d_gran, d_fault):
        rng, N = self.rng, self.noise
        # pegmatite zone: fractionation halo band, on structures, in favourable host
        halo = np.exp(-0.5 * ((d_gran - 9.0) / 5.0) ** 2)
        peg_zone = _norm(halo * (0.4 + 0.6*np.exp(-d_fault/6.0)) *
                         (0.3 + 0.7*host_schist))
        peg_zone = _norm(ndimage.gaussian_filter(peg_zone, 1.0)) * (fertile < 0.7)

        base_K = 1.0 + 1.2 * (litho == 1) + 0.6 * (litho == 2)
        K = base_K + 2.6 * peg_zone + 0.5 * fertile
        K += N * 0.35 * _grf(shape, 3, rng)

        base_Th = 6 + 10 * fertile + 4 * (litho == 2)
        Th = base_Th + 34 * peg_zone + N * 3.0 * _grf(shape, 3, rng)

        base_U = 1.5 + 3.0 * fertile
        U = base_U + 10 * peg_zone + N * 1.2 * _grf(shape, 3, rng)
        return (np.clip(K, 0, None), np.clip(Th, 0, None),
                np.clip(U, 0, None), peg_zone)

    def _magnetics(self, shape, litho, fertile, meta_mafic, peg_zone):
        rng, N = self.rng, self.noise
        tmi = (120 * meta_mafic          # mafic = magnetic high
               + 25 * fertile
               - 90 * peg_zone           # pegmatite / greisen = magnetic LOW
               + 15 * (litho == 2))
        tmi += 200 * _grf(shape, 60, rng)          # regional field
        tmi += N * 18 * _grf(shape, 2.5, rng)      # short-wavelength noise
        return tmi

    def _gravity(self, shape, litho, fertile, meta_mafic):
        rng, N = self.rng, self.noise
        g = (2.5 * meta_mafic - 1.8 * fertile)
        g += 3.0 * _grf(shape, 70, rng)
        g += N * 0.25 * _grf(shape, 4, rng)
        return g

    def _topography(self, shape, fault_mask, peg_zone):
        rng = self.rng
        dem = _fractal(shape, rng) * 320 + 400
        dem -= 40 * ndimage.gaussian_filter(fault_mask.astype(float), 2)  # fault valleys
        dem += 22 * peg_zone                              # resistant peg ridges
        return dem

    def _geochem(self, shape, peg_zone, d_fault):
        """Stream-sediment style pathfinder anomaly with down-slope dispersion."""
        rng, N = self.rng, self.noise
        anom = peg_zone.copy()
        # smear along a preferred (drainage) direction to mimic dispersion trains
        anom = ndimage.gaussian_filter(anom, sigma=(1.0, 3.5))
        bg = 0.15 + 0.1 * _grf(shape, 30, rng)
        val = _norm(anom) * (1.6 if self.commodity == "Li" else 1.4) + bg
        val += N * 0.10 * np.abs(_grf(shape, 2, rng))     # sampling noise (>=0)
        return _norm(val)

    def _plant_occurrences(self, latent):
        """Thinned inhomogeneous point process: P(keep) proportional to latent^gamma."""
        rng = self.rng
        gamma = 2.2
        n_target = self.n_occ
        intensity = latent ** gamma
        intensity /= intensity.sum()
        # sample candidate cells without replacement weighted by intensity
        flat = rng.choice(latent.size, size=min(n_target * 3, latent.size),
                          replace=False, p=intensity.ravel())
        rc = np.column_stack(np.unravel_index(flat, latent.shape))
        # spatial de-clustering: keep points >= 4 cells apart, cap at n_target
        kept = []
        for r, c in rc:
            if all((r - kr) ** 2 + (c - kc) ** 2 >= 16 for kr, kc in kept):
                kept.append((r, c))
            if len(kept) >= n_target:
                break
        kept = np.array(kept)
        # sprinkle ~8% "off-trend / historic" occurrences as label noise
        n_noise = max(1, int(0.08 * len(kept)))
        rr = rng.integers(0, latent.shape[0], n_noise)
        cc = rng.integers(0, latent.shape[1], n_noise)
        return np.vstack([kept, np.column_stack([rr, cc])])


if __name__ == "__main__":
    p = SyntheticPegmatiteProvince(commodity="Li", seed=7).generate()
    print("grid", p.shape, "| occurrences", len(p.occurrences_rc),
          "| layers", list(p.layers))
    print("latent range", round(float(p.latent.min()), 3),
          round(float(p.latent.max()), 3))
