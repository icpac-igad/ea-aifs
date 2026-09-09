#!/usr/bin/env python3
"""The N320 -> O96 regrid, shared by the inference writer and the offline transcoder.

O96 is an octahedral reduced Gaussian grid, 40,320 cells (~112 km) — the same *kind* of
grid as N320 (quasi-equal-area cells on an unstructured ``values`` axis), which is why
it drops into the existing ``(member, time, values)`` schema without any reader change.
See ``O96-icechunk-store/README.md`` for why O96 and not a 1.0 degree regular grid.

``earthkit.regrid.interpolate`` costs ~9 ms per field, most of it per-call overhead
(grid-spec lookup and validation) rather than arithmetic. Fetching the matrix once and
applying it directly cuts that to ~1-4 ms per field — verified bit-identical.

**Apply it one field at a time.** The obvious batching, ``(M @ block.T).T`` over a
whole model step, is ~9-10x *slower* than looping: the matrix is extremely sparse
(116,696 nnz in 40,320 x 542,080), and SciPy's sparse-times-dense-matrix kernel handles
that far worse than 120 sparse mat-vecs. Measured on a real 120-field step:
~4100 ms batched vs ~440 ms looped, identical results. Looping also avoids
materialising a 520 MB float64 copy of the step.
"""

import numpy as np

N_VALUES = 40320
N320_N_VALUES = 542080

_MATRIX = None


def o96_coords():
    """O96 cell centres in the point order earthkit-regrid emits.

    192 Gaussian latitude rows; row ``i`` counted from the nearest pole carries
    ``16 + 4*i`` equally spaced longitudes (``2 * sum_{i=1..96}(16+4i) = 40320``).
    Validated against ``interpolate(n320_latitudes, N320, O96)`` to < 0.006 deg, i.e.
    the ordering matches earthkit's exactly.
    """
    x, _ = np.polynomial.legendre.leggauss(192)
    lat_rows = np.degrees(np.arcsin(x))[::-1]          # north -> south
    lats, lons = [], []
    for i, la in enumerate(lat_rows, start=1):
        n = 16 + 4 * (i if i <= 96 else 193 - i)
        lats.append(np.full(n, la))
        lons.append(np.arange(n) * 360.0 / n)
    return np.concatenate(lats).astype("f4"), np.concatenate(lons).astype("f4")


def load_matrix():
    """The N320 -> O96 linear matrix earthkit itself uses (CSR, 116,696 nnz). Cached."""
    global _MATRIX
    if _MATRIX is None:
        from earthkit.regrid.db import find as db_find
        mat, out_shape = db_find({"grid": "N320"}, {"grid": "O96"}, method="linear")
        if list(out_shape) != [N_VALUES]:
            raise RuntimeError(f"unexpected O96 shape {out_shape}")
        _MATRIX = mat
    return _MATRIX


def regrid_block(block, mat=None):
    """(n_fields, 542080) -> (n_fields, 40320), one sparse mat-vec per field.

    Field by field on purpose — see the module docstring; the batched form is ~9-10x
    slower. Each field is promoted to float64 individually so the result is
    bit-identical to ``earthkit.regrid.interpolate(field, N320, O96)`` without ever
    holding a float64 copy of the whole step.
    """
    mat = load_matrix() if mat is None else mat
    block = np.asarray(block)
    out = np.empty((block.shape[0], N_VALUES), dtype="float64")
    for i in range(block.shape[0]):
        out[i] = mat @ np.asarray(block[i], dtype="float64")
    return out
