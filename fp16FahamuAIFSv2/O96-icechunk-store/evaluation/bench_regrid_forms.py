"""Why the regrid is applied per field and not batched.

The first implementation batched a whole model step into ``(M @ block.T).T``. That is
~9-10x SLOWER than looping field by field: the matrix is extremely sparse (116,696 nnz
in 40,320 x 542,080) and SciPy's sparse-times-dense-matrix kernel handles that far worse
than 120 sparse mat-vecs. The batched version starved the GPU (0% utilisation) and cost
a wasted 434 s member run. Results are bit-identical either way.

Measured on a real 120-field step from the 20260813 store:
    csr_array  batched      ~4100 ms      csr_array  per-field loop   ~440 ms
    csr_matrix batched      ~4200 ms      csr_matrix per-field loop   ~430 ms
(on an idle box the loop is ~25 ms; the ratio is what is stable.)
"""
import sys, time
sys.path.insert(0,"/home/ezra/ea-aifs/fp16FahamuAIFSv2")
import numpy as np, icechunk, zarr, scipy, scipy.sparse as sp
from earthkit.regrid.db import find as db_find
print("scipy", scipy.__version__)
B="/tank/projects/aifs-run/20260813_0000"
src=zarr.open_group(icechunk.Repository.open(icechunk.local_filesystem_storage(f"{B}/icechunk_v2")).readonly_session(tag="cycle-20260813_0000").store,mode="r")
VARS=[k for k in src.array_keys() if k not in ("latitude","longitude","time","member")]
blk=np.stack([np.asarray(src[v][0,100,:]) for v in VARS]).astype("float64")
Marr,_=db_find({"grid":"N320"},{"grid":"O96"},method="linear")   # csr_array, as shipped
Mmat=sp.csr_matrix(Marr)
one=np.repeat(blk[:1],120,axis=0)
print(f"type from db_find: {type(Marr).__name__}")

def t(f,n=3):
    f(); t0=time.time()
    for _ in range(n): r=f()
    return (time.time()-t0)/n*1000

print(f"{'form':46s} {'real data':>11s} {'repeated':>11s}")
for label, fn in [
  ("csr_array  batched (M @ X.T).T", lambda X: (Marr @ X.T).T),
  ("csr_matrix batched (M @ X.T).T", lambda X: (Mmat @ X.T).T),
  ("csr_array  per-field loop",      lambda X: np.stack([Marr @ X[i] for i in range(len(X))])),
  ("csr_matrix per-field loop",      lambda X: np.stack([Mmat @ X[i] for i in range(len(X))])),
]:
    print(f"  {label:44s} {t(lambda: fn(blk)):9.1f}ms {t(lambda: fn(one)):9.1f}ms")

ref=(Marr @ blk.T).T
loop=np.stack([Marr @ blk[i] for i in range(120)])
print(f"\nper-field loop vs batched: max|d| = {np.nanmax(np.abs(loop-ref)):.3e}")
