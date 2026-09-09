"""Is the O96 archive error small vs the ensemble spread it must not distort?"""
import numpy as np, icechunk, zarr
import earthkit.regrid as ekr
S="/tank/projects/aifs-run/20260813_0000/icechunk_v2"
repo=icechunk.Repository.open(icechunk.local_filesystem_storage(S))
r=zarr.open_group(repo.readonly_session(tag="cycle-20260813_0000").store, mode="r")
N320={"grid":"N320"}; O96={"grid":"O96"}; G15={"grid":[1.5,1.5]}
A=lambda x: ekr.interpolate(x,N320,G15)
C=lambda x: ekr.interpolate(ekr.interpolate(x,N320,O96),O96,G15)
for v in ["msl","2t","tp"]:
    for t in [95, 131]:
        a=np.stack([A(np.asarray(r[v][m,t,:]).astype("float64")) for m in range(50)])
        c=np.stack([C(np.asarray(r[v][m,t,:]).astype("float64")) for m in range(50)])
        spread=a.std(axis=0,ddof=1)                 # per-gridpoint ensemble sd
        err=np.sqrt(((c-a)**2).mean(axis=0))        # per-gridpoint regrid RMSE
        em_err=np.abs(c.mean(0)-a.mean(0))          # ensemble-mean shift
        sp_ratio=c.std(axis=0,ddof=1)/np.where(spread>0,spread,np.nan)
        print(f"{v:4s} t={t:3d} (h{(t+1)*6})  ens-spread med={np.median(spread):.4g}  "
              f"regridRMSE med={np.median(err):.4g}  ratio med={np.median(err/np.where(spread>0,spread,np.nan)):.3f}  "
              f"p95={np.nanpercentile(err/np.where(spread>0,spread,np.nan),95):.3f}  |"
              f"  ens-mean shift med={np.median(em_err):.4g}"
              f"  spread preserved med={np.nanmedian(sp_ratio):.4f}")
