"""Shared Icechunk-store readers for the MJO / tropical-storm-days diagnostics.

The per-cycle store is native N320 (`values` dim, 542 080 points), dims
`(member, time, values)`, 6-hourly, with only the downstream window written
(hours 432-792 = days 18-33); the remaining steps read back as NaN.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

N320_NPOINTS = 542080
SECONDS_PER_STEP = 21600  # 6 h

# AI-WQ tropical-storm basins (training_data.html#tropical-storm-days-data-processing)
BASINS = {
    "ATL": dict(lat=(0, 40), lon=(-100, -20)),     # North Atlantic
    "NWP": dict(lat=(0, 40), lon=(100, 180)),      # North-West Pacific
    "SWIO": dict(lat=(-40, 0), lon=(20, 90)),      # South-West Indian
    "SEIO": dict(lat=(-40, 0), lon=(90, 160)),     # South-East Indian
}


def open_store(path, tag=None, branch="main"):
    """Open an Icechunk store read-only; fall back to `branch` if `tag` missing."""
    import icechunk
    import zarr

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))
    sess = None
    if tag:
        try:
            sess = repo.readonly_session(tag=tag)
        except Exception:
            sess = None
    if sess is None:
        sess = repo.readonly_session(branch=branch)
    return zarr.open_group(sess.store, mode="r")


def coords(g):
    """(lat, lon_180, time_seconds) - lon wrapped to -180..180."""
    lat = np.asarray(g["latitude"][:], dtype=np.float64)
    lon = np.asarray(g["longitude"][:], dtype=np.float64)
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    t = np.asarray(g["time"][:], dtype=np.int64)
    return lat, lon180, t


def written_steps(g, probe_var="msl", member=0):
    """Indices of steps that actually hold data (the --write-hours window)."""
    sample = np.asarray(g[probe_var][member, :, 0])
    return np.where(~np.isnan(sample))[0]


def valid_times(g, init_date, steps):
    """datetime for each step index."""
    t = np.asarray(g["time"][:], dtype=np.int64)
    return [init_date + dt.timedelta(seconds=int(t[s])) for s in steps]


def n_members(g, var="msl"):
    return int(g[var].shape[0])


def read_member_window(g, var, member, steps, point_mask=None):
    """(len(steps), n_selected_points) float32 for one member.

    Reads the contiguous step window then subsets points in memory - far faster
    than fancy-indexing the 542 080-long last dim in zarr.
    """
    s0, s1 = int(steps[0]), int(steps[-1]) + 1
    block = np.asarray(g[var][member, s0:s1, :], dtype=np.float32)
    sel = np.asarray(steps, dtype=int) - s0
    block = block[sel]
    if point_mask is not None:
        block = block[:, point_mask]
    return block


def basin_masks(lat, lon180):
    out = {}
    for name, box in BASINS.items():
        la0, la1 = box["lat"]
        lo0, lo1 = box["lon"]
        out[name] = (lat >= la0) & (lat <= la1) & (lon180 >= lo0) & (lon180 <= lo1)
    return out


def group_steps_by_day(times):
    """[(date, [positions...]), ...] - positions index into `times`."""
    days = {}
    for i, tt in enumerate(times):
        days.setdefault(tt.date(), []).append(i)
    return sorted(days.items())


def seven_day_windows(days, start=None):
    """Consecutive non-overlapping 7-day windows over sorted dates.

    AI-WQ defines the window from the valid date forward; a cycle covering
    days 18-33 yields two full weeks (the AI-WQ week-3 / week-4 targets).
    """
    dates = [d for d, _ in days]
    if start is not None:
        dates = [d for d in dates if d >= start]
    out = []
    for i in range(0, len(dates) - 6, 7):
        out.append(dates[i:i + 7])
    return out
