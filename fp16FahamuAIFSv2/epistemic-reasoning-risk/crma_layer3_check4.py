"""Check 4: AIFS 20261001 layer 3 vs IFS 46-day op rows for the same init.

    python crma_layer3_check4.py "<crma>/layer3/ifs46/evidence_d1/op_2026_10.parquet" \
        /tank/projects/crma_layer3/20261001/evidence.parquet

Compares every region type: schema first (which region/feature keys each side
has), then per (region_type, window, feature) the ensemble means across regions --
bias, across-region correlation and spread ratio.
"""
import glob, sys
import numpy as np, pandas as pd

ifs_glob, aifs_path = sys.argv[1], sys.argv[2]
I = pd.concat([pd.read_parquet(p) for p in glob.glob(ifs_glob)])
I = I[I.init_time == "2026-10-01"]
A = pd.read_parquet(aifs_path)
for d in (I, A):
    for c in ("window", "region_id", "region_type", "feature", "node"):
        d[c] = d[c].astype(str)
print(f"IFS rows {len(I):,} (members {I.member.min()}-{I.member.max()}), "
      f"AIFS rows {len(A):,} (members {A.member.min()}-{A.member.max()})")

# ---- schema: (region_type, region_id, feature, node) keys over the windows both have
win = sorted(set(I.window) & set(A.window))
ki = set(map(tuple, I[I.window.isin(win)][["region_type", "region_id", "feature", "node"]].drop_duplicates().values))
ka = set(map(tuple, A[A.window.isin(win)][["region_type", "region_id", "feature", "node"]].drop_duplicates().values))
for rt in sorted({k[0] for k in ki | ka}):
    i_ = {k for k in ki if k[0] == rt}; a_ = {k for k in ka if k[0] == rt}
    print(f"\n[{rt}] IFS keys {len(i_)}, AIFS keys {len(a_)}, shared {len(i_ & a_)}")
    if i_ - a_:
        print("  only IFS:", sorted({(k[2], k[3]) for k in i_ - a_}), sorted({k[1] for k in i_ - a_})[:8])
    if a_ - i_:
        print("  only AIFS:", sorted({(k[2], k[3]) for k in a_ - i_}), sorted({k[1] for k in a_ - i_})[:8])

# ---- values: ensemble mean/std per key, then across regions
key = ["region_type", "window", "region_id", "feature"]
mi = I.groupby(key).value.agg(["mean", "std"]).rename(columns=lambda c: "ifs_" + c)
ma = A.groupby(key).value.agg(["mean", "std"]).rename(columns=lambda c: "aifs_" + c)
M = mi.join(ma, how="inner").reset_index().dropna(subset=["ifs_mean", "aifs_mean"])
rows = []
for (rt, w, fe), g in M.groupby(["region_type", "window", "feature"]):
    r = (np.corrcoef(g.ifs_mean, g.aifs_mean)[0, 1]
         if len(g) >= 3 and g.ifs_mean.std() > 0 and g.aifs_mean.std() > 0 else np.nan)
    rows.append({"region_type": rt, "window": w, "feature": fe, "n": len(g), "ifs": g.ifs_mean.mean(),
                 "aifs": g.aifs_mean.mean(), "bias": (g.aifs_mean - g.ifs_mean).mean(), "r": r,
                 "spread_ratio": (g.aifs_std / g.ifs_std).median()})
R = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
w = sys.argv[3] if len(sys.argv) > 3 else "D11-20"
for rt in R.region_type.unique():
    t = R[(R.region_type == rt) & (R.window == w)]
    print(f"\n== {rt} {w} ==")
    print(t.drop(columns=["region_type", "window"]).round(3).to_string(index=False))
