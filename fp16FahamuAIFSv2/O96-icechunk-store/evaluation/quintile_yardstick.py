"""Is the O96-archive perturbation large or small vs the ensemble's own sampling noise?

Reproduces the quintile calculation from shared/ensemble_quintile_analysis_cli.py, then
compares (a) N320 -> O96 -> 1.5deg vs N320 -> 1.5deg against (b) a 50-member bootstrap
resample of the N320 ensemble against the same N320 reference.
"""
import numpy as np, xarray as xr, os

B = "/tank/projects/aifs-run/20260813_0000"
WEEKS = [("week1", [0, 1], 24), ("week2", [2, 3, 4], 36)]
CLIMV = {"msl": ("mslp", "20260831", "20260907"), "tp": ("pr", "20260831", "20260907"),
         "2t": ("tas", "20260831", "20260907")}

def weekly(ncdir):
    """-> {var: {week: (50, 121, 240)}} weekly aggregate per member, CLI-identical."""
    out = {v: {} for v in CLIMV}
    per_member = []
    for m in range(1, 51):
        d = xr.open_dataset(f"{ncdir}/aifs_ensemble_forecast_1p5deg_member{m:03d}.nc")
        rec = {}
        for var in CLIMV:
            for wname, chunks, nsteps in WEEKS:
                s = sum(d[var].isel(time=t, member=0).sum(dim="step", skipna=True)
                        for t in chunks)
                rec[(var, wname)] = (s.values * 1000 * (7.0 / (nsteps * 6 / 24))
                                     if var == "tp" else s.values / nsteps)
        d.close()
        per_member.append(rec)
    for var in CLIMV:
        for wname, _, _ in WEEKS:
            out[var][wname] = np.stack([r[(var, wname)] for r in per_member])
    return out

def probs(ens, thr):
    """ens (n,121,240), thr (4,121,240) -> (5,121,240)."""
    n = ens.shape[0]
    p = np.empty((5,) + ens.shape[1:])
    p[0] = (ens < thr[0]).sum(0) / n
    for i in range(1, 4):
        p[i] = ((ens >= thr[i-1]) & (ens < thr[i])).sum(0) / n
    p[4] = (ens >= thr[3]).sum(0) / n
    return p

def thresholds():
    t = {}
    for var, (cv, d1, d2) in CLIMV.items():
        kind = "WEEKLYSUM" if cv == "pr" else "WEEKLYMEAN"
        for wname, dd in (("week1", d1), ("week2", d2)):
            f = f"{B}/aiwq/{cv}_20yrCLIM_{kind}_quintiles_{dd}.nc"
            t[(var, wname)] = xr.open_dataset(f)[cv].isel(time=0).values
    return t

def compare(P, Q, label):
    d = Q - P
    am, cm = np.argmax(P, 0), np.argmax(Q, 0)
    conf = P.max(0)
    return (f"{label:22s} RMSE={np.sqrt((d**2).mean()):.5f}  MAE={np.abs(d).mean():.5f}  "
            f"modal-flip={np.mean(am!=cm)*100:5.2f}%  (p_max>0.30: "
            f"{np.mean((am!=cm)[conf>0.30])*100:5.2f}%)")

print("loading N320 ensemble..."); A = weekly(f"{B}/nc_1p5deg")
print("loading O96  ensemble..."); C = weekly(f"{B}/nc_1p5deg_via_o96")
THR = thresholds()
ref = xr.open_dataset(f"{B}/aiwq/ensemble_quintile_probabilities_20260813_v2.nc")

rng = np.random.default_rng(0)
for var in ("msl", "2t", "tp"):
    print(f"\n=== {var} ===")
    for wi, (wname, _, _) in enumerate(WEEKS):
        thr = THR[(var, wname)]
        P = probs(A[var][wname], thr)          # production N320 reference
        Q = probs(C[var][wname], thr)          # O96 archive path
        chk = np.abs(P - ref[f"{var}_quintiles"].isel(time_week=wi).values).max()
        print(f"  {wname}  [reproduction check vs shipped file: max|d|={chk:.2e}]")
        print("   " + compare(P, Q, "O96 archive"))
        boots = [compare(P, probs(A[var][wname][rng.integers(0, 50, 50)], thr),
                         f"bootstrap #{k+1}") for k in range(3)]
        for b in boots: print("   " + b)
