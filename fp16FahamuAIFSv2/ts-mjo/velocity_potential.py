"""200 hPa velocity potential from the forecast wind field.

Why this exists
---------------
Wheeler & Hendon's RMM needs OLR, which AIFS-ENS-2.0 does not output (see
`MJO_PHASE.md`). The Velocity Potential MJO index (**VPM**, Ventrice et al.
2013) is an established RMM-like index in which **200 hPa velocity potential
replaces OLR**, leaving `[chi200, U850, U200]` in place of `[OLR, U850, U200]`.
Unlike a truncated WH04 projection, VPM is a real index with its own published
basis - and chi200 is *diagnosed exactly* from the wind field rather than
statistically emulated.

`chi` is defined by the Poisson equation on the sphere

    laplacian(chi) = D ,      D = divergence of the horizontal wind

so the whole problem is solving that for `chi` given `D`.

Why a regular grid, and why that is not a compromise
----------------------------------------------------
The store is a reduced Gaussian grid, where a Poisson solve is awkward. But VPM
consumes chi200 as a **cosine-weighted mean over +-15 deg reduced to 144
longitude points (2.5 deg)** - a planetary-scale quantity. Regridding to a
regular grid first costs nothing that survives that averaging, and turns the
solve into a textbook problem. The MJO is zonal wavenumber 1-3; nothing here is
resolution-limited at 1.5 deg.

The solver
----------
On a regular lat/lon grid the Poisson equation separates. A **real FFT in
longitude is exact** (the grid is periodic), turning `d2/dlambda2` into `-m^2`
and leaving, for each zonal wavenumber `m`, a one-dimensional ODE in latitude:

    d/dphi ( cos(phi) dchi_m/dphi )  -  m^2/cos(phi) * chi_m  =  a^2 cos(phi) D_m

which is **tridiagonal** and solved directly. Latitudes are taken at cell
centres so `cos(phi)` never vanishes, and the cosine factors at the half-levels
go to zero at the poles, which imposes the no-flux condition automatically
rather than by a boundary hack.

`m = 0` is singular - `chi` is only defined up to a constant - so that system is
pinned at one point and the area-weighted mean removed afterwards. That is not
an approximation: the constant is physically meaningless, and every downstream
step (anomaly, band mean, projection) is invariant to it.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

EARTH_RADIUS = 6371229.0    # m, IFS value


def velocity_potential(u, v, lat, lon):
    """`chi` [m^2/s] from a horizontal wind field on a regular lat/lon grid.

    Parameters
    ----------
    u, v : (..., nlat, nlon)
        Wind components. Leading axes are looped over, so a whole
        (member, time) stack can be passed at once.
    lat : (nlat,)
        Latitudes in degrees, at cell centres, monotonic. Either direction.
    lon : (nlon,)
        Longitudes in degrees, evenly spaced, spanning 360 (periodic).

    Returns
    -------
    (..., nlat, nlon) with the area-weighted mean removed on each field.
    """
    lat = np.asarray(lat, dtype=np.float64)
    flip = lat[0] > lat[-1]
    if flip:                                  # solver wants increasing latitude
        lat = lat[::-1]
        u, v = u[..., ::-1, :], v[..., ::-1, :]

    d = divergence_latlon(u, v, lat, lon)
    chi = solve_poisson_sphere(d, lat, lon)
    return chi[..., ::-1, :] if flip else chi


def divergence_latlon(u, v, lat, lon):
    """Horizontal divergence on a regular lat/lon grid.

        D = 1/(a cos(phi)) * [ du/dlambda + d(v cos(phi))/dphi ]

    Centred differences: exact and periodic in longitude via `np.roll`; the
    meridional term differentiates `v cos(phi)` as a product, so the metric term
    is carried rather than dropped.
    """
    lat = np.asarray(lat, dtype=np.float64)
    phi = np.deg2rad(lat)
    cosphi = np.cos(phi)
    dlam = np.deg2rad(float(lon[1] - lon[0]))
    dphi = float(phi[1] - phi[0])

    du = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2.0 * dlam)

    vcos = v * cosphi[:, None]
    dvcos = np.empty_like(vcos)
    dvcos[..., 1:-1, :] = (vcos[..., 2:, :] - vcos[..., :-2, :]) / (2.0 * dphi)
    # One-sided at the first/last row. v*cos(phi) -> 0 at the poles, so the
    # rows just inside them carry a small residual either way; the +-15 deg band
    # VPM uses is 75 deg away and cannot see it.
    dvcos[..., 0, :] = (vcos[..., 1, :] - vcos[..., 0, :]) / dphi
    dvcos[..., -1, :] = (vcos[..., -1, :] - vcos[..., -2, :]) / dphi

    return (du + dvcos) / (EARTH_RADIUS * cosphi[:, None])


def solve_poisson_sphere(d, lat, lon):
    """Solve `laplacian(chi) = d` on the sphere. See the module docstring."""
    lat = np.asarray(lat, dtype=np.float64)
    phi = np.deg2rad(lat)
    nlat, nlon = d.shape[-2], d.shape[-1]
    dphi = float(phi[1] - phi[0])
    cosphi = np.cos(phi)

    # cos(phi) at the half levels; zero at the poles gives no-flux for free
    ph = np.empty(nlat + 1)
    ph[1:-1] = 0.5 * (phi[:-1] + phi[1:])
    ph[0], ph[-1] = phi[0] - 0.5 * dphi, phi[-1] + 0.5 * dphi
    cos_h = np.clip(np.cos(ph), 0.0, None)
    lower_c, upper_c = cos_h[:-1] / dphi**2, cos_h[1:] / dphi**2

    lead = d.shape[:-2]
    dflat = d.reshape(-1, nlat, nlon)
    dm = np.fft.rfft(dflat, axis=-1)                 # exact: periodic in lambda
    m = np.arange(dm.shape[-1])
    chim = np.empty_like(dm)

    for k, mk in enumerate(m):
        ab = np.zeros((3, nlat))
        ab[0, 1:] = upper_c[:-1]                     # super-diagonal
        ab[2, :-1] = lower_c[1:]                     # sub-diagonal
        ab[1, :] = -(lower_c + upper_c) - (mk**2) / cosphi
        rhs = (EARTH_RADIUS**2 * cosphi)[None, :] * dm[:, :, k]
        if mk == 0:
            # singular: chi defined up to a constant. Pin one row.
            ab[1, 0], ab[0, 1] = 1.0, 0.0
            rhs = rhs.copy()
            rhs[:, 0] = 0.0
        chim[:, :, k] = (solve_banded((1, 1), ab, rhs.real.T).T
                         + 1j * solve_banded((1, 1), ab, rhs.imag.T).T)

    chi = np.fft.irfft(chim, n=nlon, axis=-1)
    w = np.broadcast_to(cosphi[:, None], (nlat, nlon))
    chi -= (chi * w).sum(axis=(-2, -1), keepdims=True) / w.sum()
    return chi.reshape(*lead, nlat, nlon)
