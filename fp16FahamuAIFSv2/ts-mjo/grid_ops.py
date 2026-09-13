"""Differential operators directly on the N320 *reduced* Gaussian grid.

The store keeps fields as a flat `values` vector (542 080 points), not a regular
lat/lon array, so `np.gradient` cannot be used and `vo` is not an output
variable. Regridding every field to a regular grid just to differentiate it
would cost ~85 ms per field (~26 min for a 50-member cycle), so instead we
exploit the reduced-Gaussian structure: the points are stored row by row, each
row is a single latitude with evenly spaced longitudes.

Relative vorticity in spherical coordinates:

    zeta = 1/(a cos(phi)) * [ dv/dlambda - d(u cos(phi))/dphi ]

d/dlambda is an exact circular difference within a row; d/dphi is taken between
the adjacent rows, matching each point to the nearest longitude in those rows
(rows have different point counts, so an index map is precomputed once).
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS = 6371229.0    # m, IFS value


class ReducedGaussianGrid:
    """Row structure of a reduced Gaussian grid + neighbour maps."""

    def __init__(self, lat, lon):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64) % 360.0
        self.lat, self.lon = lat, lon
        self.npts = lat.size

        # rows: contiguous runs of equal latitude, in storage order
        change = np.nonzero(np.diff(lat) != 0)[0] + 1
        self.row_start = np.concatenate([[0], change])
        self.row_end = np.concatenate([change, [self.npts]])
        self.nrows = self.row_start.size
        self.row_lat = lat[self.row_start]
        self.row_of = np.empty(self.npts, dtype=np.int32)
        for j, (a, b) in enumerate(zip(self.row_start, self.row_end)):
            self.row_of[a:b] = j

        # nearest-longitude neighbour in the row above / below
        self.idx_up = np.arange(self.npts)
        self.idx_dn = np.arange(self.npts)
        for j in range(self.nrows):
            a, b = self.row_start[j], self.row_end[j]
            mylon = lon[a:b]
            for tgt, arr in ((j - 1, self.idx_up), (j + 1, self.idx_dn)):
                if tgt < 0 or tgt >= self.nrows:
                    arr[a:b] = np.arange(a, b)          # edge: self (row dropped later)
                    continue
                ta, tb = self.row_start[tgt], self.row_end[tgt]
                n = tb - ta
                # rows are evenly spaced in longitude starting at 0
                k = np.rint(mylon / (360.0 / n)).astype(int) % n
                arr[a:b] = ta + k
        # rows adjacent to the poles cannot use a centred difference
        self.valid = (self.row_of > 0) & (self.row_of < self.nrows - 1)

    def _d_dlambda(self, f):
        """Centred, circular derivative along each latitude row. Exact: the row
        is a closed circle of evenly spaced longitudes."""
        out = np.empty_like(f)
        for j in range(self.nrows):
            a, b = self.row_start[j], self.row_end[j]
            seg = f[:, a:b]
            dlam = 2.0 * np.pi / (b - a)
            out[:, a:b] = (np.roll(seg, -1, axis=1) - np.roll(seg, 1, axis=1)) / (2.0 * dlam)
        return out

    def _d_dphi(self, f):
        """Derivative between adjacent rows, each point matched to the nearest
        longitude above and below via the precomputed index maps."""
        up, dn = f[:, self.idx_up], f[:, self.idx_dn]
        phi = np.deg2rad(self.lat)
        dphi = phi[self.idx_up] - phi[self.idx_dn]
        dphi = np.where(np.abs(dphi) < 1e-12, np.nan, dphi)
        return (up - dn) / dphi[None, :]

    def relative_vorticity(self, u, v):
        """(nsteps, npts) u, v  ->  (nsteps, npts) relative vorticity [1/s]."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        cosphi = np.maximum(np.cos(np.deg2rad(self.lat)), 1e-6)
        zeta = (self._d_dlambda(v) - self._d_dphi(u * cosphi[None, :])) \
            / (EARTH_RADIUS * cosphi[None, :])
        zeta[:, ~self.valid] = np.nan
        return zeta

    def divergence(self, u, v):
        """(nsteps, npts) u, v  ->  (nsteps, npts) horizontal divergence [1/s].

            D = 1/(a cos phi) * [ du/dlambda + d(v cos phi)/dphi ]

        The same two operators as `relative_vorticity` with u and v exchanged
        and the sign flipped -- deliberately sharing them, so the pair cannot
        drift apart. Feeds moisture-flux convergence and upper-level divergence
        for the evidence nodes, and the velocity potential the VPM MJO index
        needs (`velocity_potential.py`).
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        cosphi = np.maximum(np.cos(np.deg2rad(self.lat)), 1e-6)
        d = (self._d_dlambda(u) + self._d_dphi(v * cosphi[None, :])) \
            / (EARTH_RADIUS * cosphi[None, :])
        d[:, ~self.valid] = np.nan
        return d


def cyclonic(zeta, lat):
    """Sign-corrected vorticity: positive = cyclonic in both hemispheres."""
    return zeta * np.where(lat < 0, -1.0, 1.0)[None, :]
