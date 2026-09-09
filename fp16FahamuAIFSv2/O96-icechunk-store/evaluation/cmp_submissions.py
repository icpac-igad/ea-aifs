"""Compare the AI-WQ submission built from N320 vs from an O96 archive."""
import numpy as np, xarray as xr
B="/tank/projects/aifs-run/20260813_0000"
a=xr.open_dataset(f"{B}/aiwq/ensemble_quintile_probabilities_20260813_v2.nc")
c=xr.open_dataset(f"{B}/aiwq_via_o96/ensemble_quintile_probabilities_20260813_v2.nc")
print("vars:", list(a.data_vars), "| dims:", dict(a.sizes)); print()
for v in a.data_vars:
    A=a[v].values.astype("float64"); C=c[v].values.astype("float64")
    if A.shape!=C.shape: print(v,"SHAPE MISMATCH",A.shape,C.shape); continue
    m=np.isfinite(A)&np.isfinite(C)
    d=C[m]-A[m]
    # quintile axis is the leading/known dim: find it
    qax=[i for i,s in enumerate(A.shape) if s==5]
    print(f"{v:6s} n={m.sum():>9d}  RMSE={np.sqrt((d**2).mean()):.5f}  MAE={np.abs(d).mean():.5f}  "
          f"max|d|={np.abs(d).max():.4f}  bias={d.mean():+.6f}  "
          f"|d|>0.02: {(np.abs(d)>0.02).mean()*100:5.2f}%  |d|>0.05: {(np.abs(d)>0.05).mean()*100:5.2f}%")
    if qax:
        ax=qax[0]
        am=np.argmax(A,axis=ax); cm=np.argmax(C,axis=ax)
        # only where the modal category is meaningfully defined
        conf=np.max(A,axis=ax)
        print(f"       modal-quintile changes: {np.mean(am!=cm)*100:5.2f}% of gridpoints"
              f"  (where p_max>0.30: {np.mean((am!=cm)[conf>0.30])*100:5.2f}%)")
