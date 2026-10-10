"""AIFS-ENS cycle -> CRMA layer 3, so the ~256 GB cycle store can be purged.

    python crma_layer3_extract.py \
        --store /tank/projects/aifs-run/20261001_0000/icechunk_o96 --cycle 20261001 \
        --crma-repo /path/to/crma --out /tank/projects/crma_layer3

DRAFT (2026-10-10): written where /tank is not mounted, syntax-checked only.
Run it on one complete cycle and check the manifest before trusting it, or
before wiring it into the purge guard.

Why: the products are ~1e-5 of the store (CYCLE_DATA_INVENTORY.md, "Storage
arithmetic").  CRMA's evidence is the same kind of product.  Once layer 3 is
written, the O96 corpus is no longer the only source of what CRMA needs.

What it writes, per cycle, into <out>/<cycle>/ -- OUTSIDE aifs-run so
cleanup_aifs_run.py cannot reach it, like the MJO band series:

  evidence.parquet     the CRMA long layout, shared with the medium range and
                       S2S (crma/hazards/dryspell/s2s_features.py):
                       init_time, window, region_id, region_type, feature, node,
                       member (0-based, as S2S), value
                       windows D1-10 D11-20 D21-30 D31-33 (lead hours 0-792)
                       region types, each with the features s2s_features.py
                       writes for it:
                         hydrobasin  55 lev04 basins, read through a 1.5-deg
                                     buffer (the S2S 3x3 neighbourhood); the
                                     node features below
                         synoptic    regions.SYNOPTIC_LOCI; <field>_mean for
                                     every derived field, plus vo850_max,
                                     mfx850_max, slp_boxmax, slp_boxmin (daily
                                     within-box extremes, window-averaged)
                         global      regions.GLOBAL_LOCI; u200_mean u850_mean
                                     u50_mean u10_mean slp_mean
                         ocean       INDIAN_OCEAN_LOCI (node indian_ocean) and
                                     ENSO_LOCI (node enso); sst_mean, from skt
                                     over sea points (AIFS has no sst; swvl1 is
                                     NaN over sea, which gives the mask)
  tp_basins.npz        daily basin rainfall per member (mm/day), day 0..32
  regional_compact.npz the regime box and the ten gridded drivers per member
                       and window, on the S2S 1.5-deg regional grid, so the IFS
                       regime catalogues (crma/hazards/dryspell/regional_layer.py)
                       apply directly
  manifest.json        rows, members, windows, variables used and missing,
                       sha256 of each file -- the purge guard keys on it

Basin node features use the S2S names (s2s_features.FEATURES), so an AIFS row
and an S2S row are the same quantity:
  q850 q700 uq850 vq850 mfx850 slp_mean slp_max vo850 div850 div200 z500
  cape (NaN: absent in AIFS) lapse_850_500 dpd2m t2d tp
plus AIFS-only features the S2S store lacks:
  w700           omega, Pa/s, negative = ascent (vertical node)
  cp_frac        cp/tp, convective fraction
  swvl1          top-layer soil moisture (antecedent: dry spell, drought)
  ro             runoff (flow)

Derivatives use ts-mjo/grid_ops.ReducedGaussianGrid on the native O96 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ts-mjo"))
from grid_ops import ReducedGaussianGrid, cyclonic  # noqa: E402

WINDOWS = {"D1-10": (0, 240), "D11-20": (240, 480), "D21-30": (480, 720), "D31-33": (720, 792)}
FEATURES = ["q850", "q700", "uq850", "vq850", "mfx850", "slp_mean", "slp_max", "vo850",
            "div850", "div200", "z500", "cape", "lapse_850_500", "dpd2m", "t2d", "tp"]
EXTRA = ["w700", "cp_frac", "swvl1", "ro"]
S2S_FIELDS = ["q850", "q700", "uq850", "vq850", "mfx850", "slp", "vo850", "div850", "div200",
              "z500", "cape", "lapse_850_500", "dpd2m", "t2d", "tp"]   # s2s_features.derive() keys
NODE_OF = {"q850": "moisture", "q700": "moisture", "uq850": "transport", "vq850": "transport",
           "mfx850": "transport", "slp_mean": "disturbance", "slp_max": "disturbance",
           "vo850": "disturbance", "div850": "vertical", "div200": "vertical", "z500": "vertical",
           "w700": "vertical", "cape": "convective", "lapse_850_500": "convective",
           "dpd2m": "surface_humidity", "t2d": "surface_humidity", "tp": "precipitation",
           "cp_frac": "precipitation", "swvl1": "antecedent", "ro": "antecedent"}
G = 9.80665
NEEDED = ["q_850", "q_700", "u_850", "v_850", "u_200", "v_200", "z_500", "t_850", "t_500",
          "w_700", "msl", "2t", "2d", "tp", "cp", "swvl1", "ro",
          "u_50", "u_10", "skt"]          # loci: global u50/u10 (QBO), ocean sst from skt


def lead_hours(ds):
    """The lead-hour coordinate, whatever the store calls it."""
    for name in ("step", "lead_time", "time"):
        if name in ds.coords and ds[name].size > 1:
            v = ds[name].values
            if np.issubdtype(v.dtype, np.timedelta64):
                return name, (v / np.timedelta64(1, "h")).astype(int)
            if np.issubdtype(v.dtype, np.datetime64):
                return name, ((v - v[0]) / np.timedelta64(1, "h")).astype(int) + 6   # first step is 6 h
            return name, v.astype(int)
    raise SystemExit("no lead dimension found -- inspect the store")


def region_weights(lat, lon, crma):
    """Point weights per region group: {group: (ids, W (n_region, n_point))}.

    Groups follow s2s_features.py: hydrobasin (buffered basins), synoptic
    (SYNOPTIC_LOCI), global (GLOBAL_LOCI), and the two ocean groups, which S2S
    keeps apart only for their node name (indian_ocean, enso)."""
    sys.path.insert(0, os.path.join(crma, "bn-evidence"))
    sys.path.insert(0, os.path.join(crma, "medium-range-forecast", "4-bn-preprocess"))
    import ens
    import regions as reg
    import shapely
    lon180 = ((lon + 180) % 360) - 180
    cw = np.cos(np.deg2rad(lat))

    def norm(ids, rows):
        W = np.stack(rows).astype(np.float32)
        return ids, W / W.sum(1, keepdims=True)

    b = reg.basins_for_countries(4, iso3=reg.ICPAC_11, domain=ens.DOMAIN, min_area_km2=5_000.0)
    ids, rows = [], []
    for rid, geom in zip(b.region_id, b.geometry):
        m = shapely.contains_xy(geom.buffer(1.5), lon180, lat)  # the S2S 3x3 neighbourhood, ~1.5 deg
        if m.any():
            ids.append(rid); rows.append(np.where(m, cw, 0.0))
    out = {"hydrobasin": norm(ids, rows)}
    for group, boxes in (("synoptic", reg.SYNOPTIC_LOCI), ("global", reg.GLOBAL_LOCI),
                         ("indian_ocean", reg.INDIAN_OCEAN_LOCI), ("enso", reg.ENSO_LOCI)):
        ids, rows = [], []
        for name, bx in boxes.items():
            m = ((lat >= bx["lat"][0]) & (lat <= bx["lat"][1])
                 & (lon180 >= bx["lon"][0]) & (lon180 <= bx["lon"][1]))
            if m.any():
                ids.append(name); rows.append(np.where(m, cw, 0.0))
        out[group] = norm(ids, rows)
    return out


def region_mean(x, W):
    """Weighted region means of x (..., point) -> (..., region), skipping NaN points.

    A plain x @ W.T turns a whole region NaN if any one point is NaN: swvl1 and ro
    are NaN over sea and lakes (70% of points), so every coastal or lake basin and
    most loci were lost.  Here NaN points are dropped and the remaining weights
    renormalised; a region is NaN only when all of its points are."""
    ok = ~np.isnan(x)
    num = np.where(ok, x, 0.0) @ W.T
    den = ok.astype(W.dtype) @ W.T
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def nearest_index(lat, lon, glat, glon):
    """Nearest O96 point for every cell of a regular (glat, glon) grid -- for the
    regional compact fields, which must sit on the S2S 1.5-deg grid."""
    lon180 = ((lon + 180) % 360) - 180
    xyz = np.c_[np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon180)),
                np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon180)), np.sin(np.deg2rad(lat))]
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    q = np.c_[np.cos(np.deg2rad(LA.ravel())) * np.cos(np.deg2rad(LO.ravel())),
              np.cos(np.deg2rad(LA.ravel())) * np.sin(np.deg2rad(LO.ravel())), np.sin(np.deg2rad(LA.ravel()))]
    idx = np.empty(len(q), int)
    for s in range(0, len(q), 2000):                           # chunked argmax of dot product
        idx[s:s + 2000] = (q[s:s + 2000] @ xyz.T).argmax(1)
    return idx.reshape(LA.shape)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--crma-repo", required=True)
    ap.add_argument("--out", default="/tank/projects/crma_layer3")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import icechunk
    sys.path.insert(0, os.path.join(a.crma_repo, "hazards", "dryspell"))
    from regional_situation import drivers, regime_inputs      # one definition with S2S
    from s2s_features import NODE_OF as S2S_NODE_OF             # loci node names, as S2S

    ds = xr.open_zarr(icechunk.Repository.open(icechunk.local_filesystem_storage(a.store))
                      .readonly_session("main").store, consolidated=False, chunks={})
    lat, lon = ds["latitude"].values, ds["longitude"].values
    ldim, lh = lead_hours(ds)
    members = ds["member"].values
    have = [v for v in NEEDED if v in ds]
    missing = [v for v in NEEDED if v not in ds] + ["cape (absent in AIFS)", "olr (absent)"]
    print(f"{a.cycle}: {len(members)} members, {len(lh)} steps {lh.min()}-{lh.max()} h, "
          f"{lat.size} points | missing {missing}", flush=True)
    grid = ReducedGaussianGrid(lat, lon)
    groups = region_weights(lat, lon, a.crma_repo)
    bids, Wb = groups["hydrobasin"]
    sea = None                                                  # swvl1 is NaN over sea: the sea mask
    if "skt" in ds and "swvl1" in ds:
        sea = np.isnan(ds["swvl1"].isel({"member": 0, ldim: 0}).values)
    # the S2S regional grid, exactly as s2s_features.reader() cuts REGION from the
    # IFS 1.5-deg stores: lat 30 -> -39 descending (47), lon 10.5 -> 90 (54).  The
    # IFS lon nodes are multiples of 1.5, so the grid starts at 10.5, not 10.0: a
    # 10.0 start has the same shape and regime-box size and is silently 0.5 deg west.
    glat = np.arange(30.0, -40.0, -1.5); glon = np.arange(10.5, 90.0 + 1e-9, 1.5)
    assert glat.size == 47 and glon.size == 54 and glon[0] == 10.5 and glon[-1] == 90.0
    nidx = nearest_index(lat, lon, glat, glon)
    init = pd.Timestamp(a.cycle)

    def get(v, m):
        return ds[v].sel(member=m).transpose(ldim, ...).values.astype(np.float32) if v in ds else None

    def one(m):
        f = {v: get(v, m) for v in have}                        # (step, point)
        tp = f.get("tp")
        if tp is not None and np.nanmax(tp) < 1.0:             # metres per 6 h -> mm
            tp = tp * 1000.0
        q8 = f["q_850"] * 1000.0
        u8, v8 = f["u_850"], f["v_850"]
        fld = {"q850": q8, "q700": f["q_700"] * 1000.0, "uq850": u8 * q8, "vq850": v8 * q8,
               "mfx850": np.hypot(u8, v8) * q8, "slp": f["msl"] / 100.0,
               "vo850": cyclonic(grid.relative_vorticity(u8, v8), lat) * 1e5,
               "div850": grid.divergence(u8, v8) * 1e5,
               "div200": grid.divergence(f["u_200"], f["v_200"]) * 1e5,
               "z500": f["z_500"] / G, "cape": np.full_like(q8, np.nan),
               "lapse_850_500": f["t_850"] - f["t_500"], "dpd2m": f["2t"] - f["2d"], "t2d": f["2d"],
               "tp": tp * 4.0 if tp is not None else None,      # 6-h interval -> mm/day rate
               "w700": f.get("w_700"),
               "cp": f.get("cp"),
               "swvl1": f.get("swvl1"), "ro": f.get("ro")}
        recs = []                                   # (window, region_type, ids, feature, node, values)

        def daily(x, sel):                          # 6-hourly (step, point) -> daily means (day, point)
            return x[sel].reshape(-1, 4, x.shape[1]).mean(1)

        for wname, (h0, h1) in WINDOWS.items():
            sel = (lh > h0) & (lh <= h1)
            if not sel.any():
                continue
            assert sel.sum() % 4 == 0, f"{wname}: {sel.sum()} steps is not whole days"
            # -- basins: the 16 S2S node features plus the AIFS-only extras
            for fe, x in fld.items():
                if x is None:
                    continue
                if fe == "slp":
                    rb = region_mean(daily(x, sel), Wb)                          # (day, basin)
                    recs.append((wname, "hydrobasin", bids, "slp_mean", NODE_OF["slp_mean"], rb.mean(0)))
                    recs.append((wname, "hydrobasin", bids, "slp_max", NODE_OF["slp_max"], rb.max(0)))
                    continue
                if fe == "cp":
                    # convective fraction = window-total cp / window-total tp, per basin
                    # (a mean of point ratios is undefined where it does not rain)
                    if f.get("tp") is not None:
                        num = np.nansum(x[sel], 0) @ Wb.T
                        den = np.nansum(f["tp"][sel], 0) @ Wb.T
                        recs.append((wname, "hydrobasin", bids, "cp_frac", NODE_OF["cp_frac"],
                                     np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)))
                    continue
                recs.append((wname, "hydrobasin", bids, fe, NODE_OF[fe], region_mean(np.nanmean(x[sel], 0), Wb)))
            # -- synoptic loci, as s2s_features.py: <field>_mean, plus daily box extremes
            sids, Ws = groups["synoptic"]
            for fe in S2S_FIELDS:
                x = fld[fe]
                if x is None:
                    continue
                node = "precipitation" if fe == "tp" else S2S_NODE_OF.get(fe, S2S_NODE_OF.get(fe + "_mean", "disturbance"))
                recs.append((wname, "synoptic", sids, f"{fe}_mean", node, region_mean(daily(x, sel), Ws).mean(0)))
            for name, fe, sign, node in (("vo850_max", "vo850", 1, "disturbance"),
                                         ("mfx850_max", "mfx850", 1, "transport"),
                                         ("slp_boxmax", "slp", 1, "disturbance"),
                                         ("slp_boxmin", "slp", -1, "disturbance")):
                dx = daily(fld[fe], sel) * sign                                  # (day, point)
                bm = np.stack([np.nanmax(dx[:, w > 0], 1) for w in Ws], 1)       # (day, box)
                recs.append((wname, "synoptic", sids, name, node, sign * bm.mean(0)))
            # -- global loci: u200, u850, u50, u10, slp
            gids, Wg = groups["global"]
            for name, x in (("u200_mean", f["u_200"]), ("u850_mean", u8), ("u50_mean", f.get("u_50")),
                            ("u10_mean", f.get("u_10")), ("slp_mean", fld["slp"])):
                if x is not None:
                    recs.append((wname, "global", gids, name, "global", region_mean(np.nanmean(x[sel], 0), Wg)))
            # -- ocean loci: sst_mean from skt over sea points (skt over sea is the SST)
            if sea is not None:
                sst = np.where(sea[None], f["skt"], np.nan)
                for group in ("indian_ocean", "enso"):
                    oids, Wo = groups[group]
                    recs.append((wname, "ocean", oids, "sst_mean", group, region_mean(np.nanmean(sst[sel], 0), Wo)))
        ev = pd.concat([pd.DataFrame({"window": w, "region_id": list(rids), "region_type": rt,
                                      "feature": fe, "node": nd, "value": np.asarray(v, np.float32)})
                        for w, rt, rids, fe, nd, v in recs], ignore_index=True)
        ev["member"] = int(m) - 1                    # 0-based, as S2S (store member 1 -> 0)
        # daily basin rainfall
        tpd = None
        if tp is not None:
            nd = len(lh) // 4
            tpd = region_mean(tp[:nd * 4].reshape(nd, 4, -1).sum(1), Wb)    # (day, basin) mm/day
        # regional compact, on the S2S 1.5-deg grid, window means
        reg_vars = {"u850": u8, "v850": v8, "q850": q8, "slp": f["msl"] / 100.0,
                    "olr": np.full_like(u8, np.nan), "u200": f["u_200"], "z500": f["z_500"] / G,
                    "tp": tp * 4.0 if tp is not None else np.full_like(u8, np.nan)}
        Fw = []
        for wname, (h0, h1) in list(WINDOWS.items())[:3]:
            sel = (lh > h0) & (lh <= h1)
            Fw.append(np.stack([np.nanmean(reg_vars[v][sel], 0)[nidx] for v in
                                ["u850", "v850", "q850", "slp", "olr", "u200", "z500", "tp"]]))
        Fw = np.stack(Fw)[None]                                   # (1, window, var, lat, lon)
        vars8 = ["u850", "v850", "q850", "slp", "olr", "u200", "z500", "tp"]
        return m, ev, tpd, regime_inputs(Fw, glat, glon, vars8)[0], drivers(Fw, glat, glon, vars8)[0]

    od = os.path.join(a.out, a.cycle); os.makedirs(od, exist_ok=True)
    evs, tps, Xs, Ds = [], {}, {}, {}
    with ThreadPoolExecutor(a.workers) as ex:
        for m, ev, tpd, X, D in ex.map(one, members):
            evs.append(ev); tps[m] = tpd; Xs[m] = X; Ds[m] = D
            print(f"  member {m} done", flush=True)
    E = pd.concat(evs, ignore_index=True)
    E.insert(0, "init_time", init)
    for c in ("window", "region_id", "region_type", "feature", "node"):
        E[c] = E[c].astype("category")
    p_ev = os.path.join(od, "evidence.parquet"); E.to_parquet(p_ev, index=False)
    ms = sorted(tps)
    p_tp = os.path.join(od, "tp_basins.npz")
    np.savez_compressed(p_tp, tp=np.stack([tps[m] for m in ms]).astype(np.float32),
                        members=np.array(ms) - 1, ids=np.array(bids), init=str(init.date()))
    p_rc = os.path.join(od, "regional_compact.npz")
    np.savez_compressed(p_rc, X=np.stack([Xs[m] for m in ms])[None].astype(np.float16),
                        DRV=np.stack([Ds[m] for m in ms])[None].astype(np.float32),
                        inits=np.array([init.to_datetime64()]).astype("datetime64[D]"),
                        windows=np.array(["D1-10", "D11-20", "D21-30"]), lat=glat, lon=glon)
    man = {"cycle": a.cycle, "store": a.store, "members": len(ms), "steps": int(len(lh)),
           "windows": list(WINDOWS), "rows": int(len(E)),
           "regions": {k: len(v[0]) for k, v in groups.items()}, "member_base": 0,
           "sst_from": "skt over sea (swvl1 NaN)" if sea is not None else None,
           "variables_used": have, "missing": missing, "draft": True,
           "files": {os.path.basename(p): {"bytes": os.path.getsize(p), "sha256": sha(p)}
                     for p in (p_ev, p_tp, p_rc)}}
    json.dump(man, open(os.path.join(od, "manifest.json"), "w"), indent=1)
    print(f"wrote {od}: {len(E):,} evidence rows, {len(ms)} members -- manifest.json")


if __name__ == "__main__":
    main()
