"""Checks 1-3 of CRMA_LAYER3.md on each cycle's layer-3 output.

    python crma_layer3_check.py 20260903 20260910 ...
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd, xarray as xr, icechunk

EXPECTED_NAN = {("cape", "hydrobasin"), ("cape_mean", "synoptic"),   # no CAPE in AIFS
                ("cp_frac", "hydrobasin")}                          # no rain in the window

def op(c, p):
    r = icechunk.Repository.open(icechunk.local_filesystem_storage(f"/tank/projects/aifs-run/{c}_0000/{p}"))
    return xr.open_zarr(r.readonly_session("main").store, consolidated=False, chunks={})

for c in sys.argv[1:]:
    D = f"/tank/projects/crma_layer3/{c}/"
    m = json.load(open(D + "manifest.json"))
    sha_ok = all(hashlib.sha256(open(D + f, "rb").read()).hexdigest() == v["sha256"] for f, v in m["files"].items())
    size = sum(v["bytes"] for v in m["files"].values()) / 1e6
    E = pd.read_parquet(D + "evidence.parquet")
    nreg = E.groupby("region_type", observed=True).region_id.nunique()
    cnt = E.groupby(["window", "region_type", "feature"], observed=True).size()
    full = (cnt == cnt.index.get_level_values("region_type").astype(str).map(nreg).to_numpy() * m["members"]).all()
    nan = E.assign(n=E.value.isna()).groupby(["feature", "region_type"], observed=True).n.mean()
    unexpected = {k: round(v, 3) for k, v in nan[nan > 0].items() if k not in EXPECTED_NAN}
    c1 = full and not unexpected and len(E) == m["rows"]
    # check 2: O96 vs N320 sidecar tp, EA box, h432-792. Older cycles have the N320 store
    # under another name, or none at all (then C2 is n/a and only the units rule applies).
    side = next((p for p in ("icechunk_n320_aiwq", "icechunk_n320_nwh")
                 if os.path.isdir(f"/tank/projects/aifs-run/{c}_0000/{p}")), None)
    ratios = []
    for mem in ((1, 26, 50) if side else ()):
        vals = []
        for p in ("icechunk_o96", side):
            ds = op(c, p); la, lo = ds.latitude.values, ds.longitude.values
            bx = (la > -12) & (la < 15) & (lo > 22) & (lo < 52)
            vals.append(float(ds.tp.sel(member=mem).isel(time=slice(71, 132)).values[:, bx].mean()))
        ratios.append(vals[0] / vals[1])
    c2 = all(abs(r - 1) < 0.05 for r in ratios) if side else None
    # check 3: lead hours
    t = op(c, "icechunk_o96").time.values
    lh = ((t - np.datetime64(pd.Timestamp(c))) / np.timedelta64(1, "h")).astype(int)
    c3 = len(lh) == 132 and lh[0] == 6 and lh[-1] == 792 and set(np.diff(lh)) == {6}
    s2 = (f"C2 {'PASS' if c2 else 'FAIL'} O96/{side} {[round(r, 3) for r in ratios]}" if side
          else "C2 n/a (no N320 store)")
    print(f"{c}: members {m['members']} steps {m['steps']} rows {m['rows']:,} regions {dict(nreg)} "
          f"{size:.1f} MB sha {'OK' if sha_ok else 'MISMATCH'} | "
          f"C1 {'PASS' if c1 else 'FAIL'}{' unexpected NaN ' + str(unexpected) if unexpected else ''} | "
          f"{s2} | C3 {'PASS' if c3 else 'FAIL'} {lh[0]}-{lh[-1]} h", flush=True)
