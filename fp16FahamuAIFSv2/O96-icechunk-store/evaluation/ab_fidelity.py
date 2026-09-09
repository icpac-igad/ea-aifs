"""A/B fidelity: N320->1.5deg (production) vs N320->O96->1.5deg (proposed archive)."""
import numpy as np, icechunk, zarr, time
import earthkit.regrid as ekr

S="/tank/projects/aifs-run/20260813_0000/icechunk_v2"
repo=icechunk.Repository.open(icechunk.local_filesystem_storage(S))
r=zarr.open_group(repo.readonly_session(tag="cycle-20260813_0000").store, mode="r")

N320={"grid":"N320"}; O96={"grid":"O96"}; G15={"grid":[1.5,1.5]}; G10={"grid":[1.0,1.0]}
A=lambda x: ekr.interpolate(x,N320,G15)
C=lambda x: ekr.interpolate(ekr.interpolate(x,N320,O96),O96,G15)

# also: what does a 1.0deg archive cost via scipy bilinear second hop?
from scipy.interpolate import RegularGridInterpolator
lat10=np.linspace(90,-90,181); lon10=np.arange(0,360,1.0)
lat15=np.linspace(90,-90,121); lon15=np.arange(0,360,1.5)
LO,LA=np.meshgrid(lon15,lat15)
pts=np.stack([LA.ravel(),LO.ravel()],axis=-1)
def B(x):
    g=ekr.interpolate(x,N320,G10)
    gx=np.concatenate([g,g[:,:1]],axis=1)                 # wrap lon 360
    f=RegularGridInterpolator((lat10[::-1],np.append(lon10,360.0)),gx[::-1,:],
                              bounds_error=False,fill_value=None)
    return f(pts).reshape(121,240)

VARS=["msl","2t","tp"]
MEMBERS=[0,12,25,37,49]
TIDX=[77,95,113,131]        # h468, 576, 684, 792
print(f"{'var':5s} {'path':6s} {'RMSE':>10s} {'MAE':>10s} {'maxabs':>10s} {'bias':>11s} {'field-sd':>10s} {'RMSE/sd':>8s}")
for v in VARS:
    accB={'se':0.,'ae':0.,'mx':0.,'b':0.,'n':0,'var':0.}
    accC={'se':0.,'ae':0.,'mx':0.,'b':0.,'n':0,'var':0.}
    for m in MEMBERS:
        for t in TIDX:
            x=np.asarray(r[v][m,t,:]).astype("float64")
            a=A(x); b=B(x); c=C(x)
            for acc,y in ((accB,b),(accC,c)):
                d=y-a
                acc['se']+=float((d**2).sum()); acc['ae']+=float(np.abs(d).sum())
                acc['mx']=max(acc['mx'],float(np.abs(d).max())); acc['b']+=float(d.sum())
                acc['n']+=d.size; acc['var']+=float(((a-a.mean())**2).sum())
    for lbl,acc in (("1.0deg",accB),("O96",accC)):
        n=acc['n']; sd=(acc['var']/n)**0.5
        rmse=(acc['se']/n)**0.5
        print(f"{v:5s} {lbl:6s} {rmse:10.4g} {acc['ae']/n:10.4g} {acc['mx']:10.4g} "
              f"{acc['b']/n:11.4g} {sd:10.4g} {rmse/sd*100:7.3f}%")
