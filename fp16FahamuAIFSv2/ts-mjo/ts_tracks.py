"""Tropical-cyclone tracking on the N320 reduced Gaussian grid.

Why this module exists
----------------------
AI-WQ's observed target counts **unique `(storm, day)` pairs**
(`AI_WQ_package.TS_processing.count_nstormdays`): two concurrent hurricanes in
the Atlantic on one day contribute *two* storm days, so a week is unbounded
above (observed maximum 28 in the 20-year sample). A point-wise detector that
asks "did any gridpoint in the basin qualify today?" returns a boolean and is
capped at 7 per week. Those are different quantities, and no threshold tuning
reconciles them - the counter has to carry **storm identity**.

That is what this module adds:

1. **centres** - per step, cyclonic low-pressure candidates are reduced to
   discrete centres by greedy lowest-pressure-first picking with a minimum
   separation (one system, one centre, not a blob of points), and each centre
   is then tested against *its own surroundings* for a closed pressure minimum
   and a 200-500 hPa warm core;
2. **tracks** - centres are linked across consecutive 6-hourly steps by
   nearest-neighbour matching bounded by a maximum translation speed, and
   short-lived matches are dropped;
3. **counting** - `(track_id, day)` pairs, filtered per record by TS strength
   and by the centre lying inside the basin box, exactly mirroring
   `count_nstormdays`'s `TS_level & loc_mask` on a per-record basis.

Wind is treated the way IBTrACS treats it
-----------------------------------------
`usa_wind` is the storm's *maximum sustained wind*, not the wind at the centre
- and in a resolved cyclone the pressure minimum and the wind maximum are never
the same gridpoint (the eyewall is 30-100 km out). The earlier detector
required wind, pressure, warmth and vorticity to co-occur **at one point**,
which is physically unsatisfiable for a well-resolved storm and noise-prone for
a marginal one. Here the centre is located from the core fields, and the wind
test is `max(|V10|) within --wind-radius-km of the centre >= threshold`, which
is the standard tracker formulation and the like-for-like match to `usa_wind`.

Sub-TS records are kept in the track for continuity and excluded from the
count, again mirroring IBTrACS: a storm that weakens below 34 kt and re-intensifies
is one storm, and only its at-strength days are counted.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.229
KM_PER_DEG_LAT = 111.195

# defaults - all exposed as CLI flags in ts_days.py
CENTRE_SEPARATION_KM = 500.0   # two centres closer than this are one system
WIND_RADIUS_KM = 300.0         # search radius for the storm's maximum wind
CORE_RADIUS_KM = 200.0         # inner disc averaged for the warm-core test
ENV_RADIUS_KM = 800.0          # outer radius; the annulus is the environment
WARM_CORE_LEVELS = ("t_200", "t_300", "t_500")   # 200-500 hPa layer mean
MAX_TRANSLATION_MS = 25.0      # m/s cap on centre displacement between steps
MIN_TRACK_HOURS = 24.0         # a coherent low must persist this long to count


def haversine_km(lat0, lon0, lat, lon):
    """Great-circle distance [km] from one point to an array of points."""
    p0, p1 = np.deg2rad(lat0), np.deg2rad(lat)
    dphi = p1 - p0
    dlam = np.deg2rad(lon - lon0)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class PointIndex:
    """Radius queries on the flat `values` vector.

    A Gaussian grid is stored row by row from the north pole southwards, so
    latitude is non-increasing along the vector and any latitude band is a
    *contiguous* slice. That turns a radius query into a `searchsorted` for the
    band, a longitude window inside it, and a haversine on what survives -
    which keeps the per-centre wind search at a few thousand points instead of
    542 080.
    """

    def __init__(self, lat, lon180):
        self.lat = np.asarray(lat, dtype=np.float64)
        self.lon = np.asarray(lon180, dtype=np.float64)
        if np.any(np.diff(self.lat) > 1e-9):
            raise ValueError("latitude is not non-increasing in storage order; "
                             "PointIndex assumes Gaussian row order")
        self._neg_lat = -self.lat          # ascending, for searchsorted

    def within(self, lat0, lon0, radius_km, with_distance=False):
        """Indices of grid points within `radius_km` of (lat0, lon0)."""
        empty = np.empty(0, dtype=np.int64)
        dlat = radius_km / KM_PER_DEG_LAT
        i0 = int(np.searchsorted(self._neg_lat, -(lat0 + dlat), side="left"))
        i1 = int(np.searchsorted(self._neg_lat, -(lat0 - dlat), side="right"))
        if i1 <= i0:
            return (empty, empty.astype(float)) if with_distance else empty
        coslat = max(np.cos(np.deg2rad(lat0)), 0.05)
        dlon = min(radius_km / (KM_PER_DEG_LAT * coslat), 180.0)
        sub_lon = self.lon[i0:i1]
        near = np.abs(((sub_lon - lon0 + 180.0) % 360.0) - 180.0) <= dlon
        idx = np.nonzero(near)[0] + i0
        if idx.size == 0:
            return (idx, idx.astype(float)) if with_distance else idx
        d = haversine_km(lat0, lon0, self.lat[idx], self.lon[idx])
        keep = d <= radius_km
        return (idx[keep], d[keep]) if with_distance else idx[keep]


def band_median(field, lat, bin_deg=2.0):
    """Per-step median of `field` within each latitude band, broadcast back.

    Used only for the cheap candidate net in `detect_centres`: the detection
    belt spans tropics to midlatitudes, where mean sea-level pressure has a
    large meridional gradient, so a whole-belt median would flag every
    midlatitude low as deep and miss shallow tropical ones. The criteria that
    actually decide a centre are local to that centre, not banded.
    """
    bins = np.floor(np.asarray(lat) / bin_deg).astype(int)
    ref = np.empty_like(field)
    for b in np.unique(bins):
        cols = bins == b
        ref[:, cols] = np.nanmedian(field[:, cols], axis=1, keepdims=True)
    return ref


def detect_centres(msl, czeta, tlayer, spd, lat, lon180, belt, pidx, *,
                   msl_drop, vort_thr, warm_core_k, sep_km=CENTRE_SEPARATION_KM,
                   wind_radius_km=WIND_RADIUS_KM, core_radius_km=CORE_RADIUS_KM,
                   env_radius_km=ENV_RADIUS_KM, band_deg=2.0):
    """Per-step warm-core cyclone centres.

    Two stages, because the discriminating tests are the expensive ones:

    **Prefilter** (vectorised over the detection belt) - cyclonic 850 hPa
    vorticity above `vort_thr`, and a pressure below the latitude-band median by
    `msl_drop`. This is only a candidate net; on its own it catches every
    midlatitude low in the belt.

    **Per-centre tests** - candidates are reduced to centres by greedy
    lowest-pressure-first picking with a `sep_km` minimum separation, then each
    centre is tested against *its own surroundings* rather than against a zonal
    reference:

    - **warm core**: the 200-500 hPa layer-mean temperature averaged within
      `core_radius_km` must exceed the mean over the surrounding annulus
      (out to `env_radius_km`) by `warm_core_k`. This is the criterion that
      separates a tropical cyclone from an extratropical one, which is
      cold-core aloft; a single-level anomaly against a latitude-band median
      does not, because a midlatitude low sitting in a warm airmass passes it.
    - **pressure deficit**: the centre must be `msl_drop` below the annulus
      mean, i.e. a genuinely closed local low rather than a point on a
      large-scale gradient.
    - **wind**: `max |V10|` within `wind_radius_km`, recorded but not filtered
      on - the TS-strength cut happens at counting time so that a weakening
      storm stays linked to its own track.

    Returns `[[(plat, plon, pmsl, vmax), ...], ...]`, one list per step.
    """
    nsteps = msl.shape[0]
    lat_b, lon_b = lat[belt], lon180[belt]

    msl_b = msl[:, belt]
    cand_mask = msl_b <= band_median(msl_b, lat_b, band_deg) - msl_drop
    if vort_thr:
        cand_mask &= np.nan_to_num(czeta[:, belt], nan=-1.0) >= vort_thr

    out = []
    for s in range(nsteps):
        cand = np.nonzero(cand_mask[s])[0]
        if cand.size == 0:
            out.append([])
            continue
        # deepest first: the pressure minimum defines the centre, and anything
        # within sep_km of an accepted centre is the same system's skirt
        cand = cand[np.argsort(msl_b[s, cand], kind="stable")]
        c_lat, c_lon = lat_b[cand], lon_b[cand]
        taken = np.zeros(cand.size, dtype=bool)
        centres = []
        for k in range(cand.size):
            if taken[k]:
                continue
            plat, plon = float(c_lat[k]), float(c_lon[k])
            rest = ~taken
            rest[:k + 1] = False
            if rest.any():
                d = haversine_km(plat, plon, c_lat[rest], c_lon[rest])
                taken[np.nonzero(rest)[0][d <= sep_km]] = True

            # one radius query serves all three per-centre tests
            idx, dist = pidx.within(plat, plon, env_radius_km, with_distance=True)
            if idx.size == 0:
                continue
            inner = dist <= core_radius_km
            annulus = ~inner
            if not inner.any() or not annulus.any():
                continue
            pc = float(msl_b[s, cand[k]])
            if pc > float(np.nanmean(msl[s, idx[annulus]])) - msl_drop:
                continue
            if warm_core_k:
                dt_core = (float(np.nanmean(tlayer[s, idx[inner]]))
                           - float(np.nanmean(tlayer[s, idx[annulus]])))
                if not (dt_core >= warm_core_k):
                    continue
            near = idx[dist <= wind_radius_km]
            vmax = float(np.nanmax(spd[s, near])) if near.size else float("nan")
            centres.append((plat, plon, pc, vmax))
        out.append(centres)
    return out


def link_tracks(centres_per_step, step_hours=6.0,
                max_speed_ms=MAX_TRANSLATION_MS):
    """Link per-step centres into tracks by bounded nearest-neighbour matching.

    Greedy on ascending distance and one-to-one, so a split or a merge costs the
    weaker pairing rather than duplicating a track. The displacement cap is a
    translation speed: 25 m/s over 6 h is ~540 km, comfortably above observed TC
    motion including recurving storms, and tight enough that two unrelated lows
    in the same basin are not chained together.
    """
    max_disp_km = max_speed_ms * step_hours * 3600.0 / 1000.0
    tracks = []               # list of list[(step, lat, lon, msl, vmax)]
    active = []               # indices into `tracks` extended at the last step

    def _start(step, centre_list):
        started = []
        for c in centre_list:
            tracks.append([(step, *c)])
            started.append(len(tracks) - 1)
        return started

    for s, centres in enumerate(centres_per_step):
        if not centres:
            active = []
            continue
        if not active:
            active = _start(s, centres)
            continue

        prev_pts = np.array([[tracks[ti][-1][1], tracks[ti][-1][2]] for ti in active])
        cur_pts = np.array([[c[0], c[1]] for c in centres])
        d = np.stack([haversine_km(p[0], p[1], cur_pts[:, 0], cur_pts[:, 1])
                      for p in prev_pts])                      # (nprev, ncur)

        used_prev, used_cur, new_active = set(), set(), []
        for flat in np.argsort(d, axis=None):
            pi, ci = np.unravel_index(flat, d.shape)
            if d[pi, ci] > max_disp_km:
                break
            if pi in used_prev or ci in used_cur:
                continue
            used_prev.add(int(pi))
            used_cur.add(int(ci))
            ti = active[pi]
            tracks[ti].append((s, *centres[ci]))
            new_active.append(ti)
        unmatched = [c for ci, c in enumerate(centres) if ci not in used_cur]
        active = new_active + _start(s, unmatched)
    return tracks


def in_basin(plat, plon, box):
    la0, la1 = box["lat"]
    lo0, lo1 = box["lon"]
    return (la0 <= plat <= la1) and (lo0 <= plon <= lo1)


def storm_day_counts(tracks, times, basins, *, wind_thr, step_hours=6.0,
                     min_track_hours=MIN_TRACK_HOURS):
    """`{basin: {date: n}}` where n = unique tracks at TS strength in the box.

    This is `count_nstormdays` with `storm_ids` supplied by the tracker: unique
    `(track, day)` pairs, each record screened by TS strength and by the centre
    being inside the basin - so two concurrent storms on one day count twice and
    the weekly total is unbounded, as the target is.
    """
    min_records = max(1, int(round(min_track_hours / step_hours)) + 1)
    counts = {b: {} for b in basins}
    kept = 0
    for tid, tr in enumerate(tracks):
        span_h = (tr[-1][0] - tr[0][0]) * step_hours
        if len(tr) < min_records or span_h < min_track_hours:
            continue
        kept += 1
        for (s, plat, plon, _pmsl, vmax) in tr:
            if not (np.isfinite(vmax) and vmax >= wind_thr):
                continue
            day = times[s].date()
            for b, box in basins.items():
                if in_basin(plat, plon, box):
                    counts[b].setdefault(day, set()).add(tid)
    return {b: {d: len(v) for d, v in dd.items()} for b, dd in counts.items()}, kept
