"""Measure the rotation taking a VPM (VPM1, VPM2) pair onto RMM's convention.

Why this is needed
------------------
AI-WQ scores the MJO product against **RMM** (`AI_WQ_package` ships
`WH04_combinedEOFs.nc` and `WH04_RMM_stddevs.nc`, and `retrieve_daily_MJO_obs`
returns RMM `phase`/`amplitude`). This project produces **VPM**, because the
model has no OLR. The two indices do not share a phase convention, and the gap
is not small: submitting VPM phases as RMM phases is wrong by roughly **four
octants**, systematically -- which scores worse than climatology while looking
like a modelling failure rather than a bookkeeping one.

Method
------
Both indices are 2-D time series. The best rigid rotation between them is the
orthogonal Procrustes solution, `argmin_R ||R V - M||`, obtained from the SVD of
`M^T V`. Using a *rotation* rather than fitting each component separately is
deliberate: it cannot rescale, so it can only fix the convention, and the
residual correlation is then an honest measure of how well the two indices agree
once the convention is removed.

The components are correlated directly rather than the derived phases, because
phases hide a sign flip: with the flip in place the octant agreement is 11.5%,
which reads as "these indices are unrelated" when in fact `r = -0.804`.

*** WHICH RMM THIS IS ***
The default source is PSL's `rmm_star_data.txt` -- PSL's own RMM realisation, not
BOM's official RMM, which is most likely what AI-WQ distributes. The *existence*
and rough *size* of the offset are established by this; the exact value for the
official series is **not**, and must be re-measured with
`retrieve_daily_MJO_obs()` before anything is submitted. Pass `--rmm` to point at
a different series.
"""
from __future__ import annotations

import argparse

import numpy as np


def load_psl(path, cols=(4, 5, 6)):
    """PSL index text: year month day hour PC1 PC2 amplitude. -99 is missing."""
    a = np.loadtxt(path)
    d = np.array([np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(dd):02d}")
                  for y, m, dd in a[:, :3]])
    ok = a[:, cols[0]] > -99
    return d[ok], a[ok][:, list(cols)]


def octant(pcs, amp):
    o = (np.floor((np.arctan2(pcs[:, 1], pcs[:, 0]) + np.pi) / (np.pi / 4)).astype(int) % 8) + 1
    return np.where(amp < 1, 0, o)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpm", default="/tank/projects/era5_vpm_clim/vpm.1x.txt")
    ap.add_argument("--rmm", default="/tank/projects/era5_vpm_clim/rmm_star_data.txt")
    ap.add_argument("--out", default="/tank/projects/era5_vpm_clim/vpm_to_rmm_rotation.npz")
    args = ap.parse_args()

    dv, V = load_psl(args.vpm)
    dr, M = load_psl(args.rmm)
    c = np.intersect1d(dv, dr)
    V = V[np.searchsorted(dv, c)]
    M = M[np.searchsorted(dr, c)]
    print(f"{len(c)} common days, {c[0]} .. {c[-1]}")

    # The target series need not be unit-variance; put it on VPM's scale so the
    # `amplitude < 1` active test means the same thing on both sides.
    scale = float(np.sqrt((M[:, :2] ** 2).mean() / (V[:, :2] ** 2).mean()))
    Mn = M[:, :2] / scale
    Ma = np.hypot(*Mn.T)
    print(f"  target rescaled by 1/{scale:.4f} -> mean amp {Ma.mean():.2f}, "
          f"P(<1)={np.mean(Ma < 1):.2f}   (VPM {V[:, 2].mean():.2f}, {np.mean(V[:, 2] < 1):.2f})")
    print("  component correlations BEFORE rotation: "
          + ", ".join(f"PC{i+1} r={np.corrcoef(V[:, i], Mn[:, i])[0, 1]:+.3f}" for i in range(2)))

    U, _, Wt = np.linalg.svd(Mn.T @ V[:, :2])
    R = U @ Wt
    det = float(np.linalg.det(R))
    # A REFLECTION (det = -1) is the right answer here and must not be suppressed.
    # The two PSL files carry opposite sign conventions on one component, which
    # shows up physically: on active consecutive days the phase angle advances
    #     vpm.1x.txt      -6.81 deg/day   WESTWARD
    #     rmm_star.txt    +6.79 deg/day   EASTWARD
    # Same magnitude (360/6.8 = 53 days, squarely the MJO period), opposite sign.
    # An MJO index must advance eastward through phases 1..8, so `vpm.1x.txt` is
    # the mirrored one -- and because the basis in `vpm_basis_fit.py` was fitted
    # against that file, OUR index inherits the mirror too. Forcing det=+1 here
    # "fixes" the matrix into a pure rotation and destroys the agreement
    # (+36 deg, 15.9% same octant, against -1 reflection giving 64.0%).
    ang = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    Vr = V[:, :2] @ R.T
    kind = "REFLECTION (mirror + rotate)" if det < 0 else "rotation"
    print(f"\n  {kind}: angle {ang:+.2f} deg ({ang/45:+.2f} octants), det={det:+.3f}")
    print("  component correlations AFTER: "
          + ", ".join(f"PC{i+1} r={np.corrcoef(Vr[:, i], Mn[:, i])[0, 1]:+.3f}" for i in range(2)))

    both = (V[:, 2] >= 1) & (Ma >= 1)
    for tag, P in (("before", V[:, :2]), ("after ", Vr)):
        pv, pm = octant(P, V[:, 2]), octant(Mn, Ma)
        dd = (np.degrees(np.arctan2(P[both, 1], P[both, 0])
                         - np.arctan2(Mn[both, 1], Mn[both, 0])) + 180) % 360 - 180
        print(f"  {tag}: same octant {100*np.mean(pv[both]==pm[both]):5.1f}%  "
              f"within +-1 {100*np.mean(np.abs(((pv[both]-pm[both]+3)%8)-3)<=1):5.1f}%  "
              f"median |err| {np.median(np.abs(dd)):5.1f} deg  "
              f"| all-day 9-cat match {100*np.mean(pv==pm):5.1f}%")

    # Physical acceptance test: after the transform the index MUST advance
    # eastward. This is what makes the reflection safe to apply rather than a
    # number that merely improved a correlation.
    def drift(pcs, amp, dates):
        a = np.unwrap(np.arctan2(pcs[:, 1], pcs[:, 0]))
        step = np.diff(dates).astype(int)
        m = (step == 1) & (amp[:-1] >= 1) & (amp[1:] >= 1)
        return float(np.median(np.degrees(np.diff(a))[m]))
    d_before = drift(V[:, :2], V[:, 2], c)
    d_after = drift(Vr, V[:, 2], c)
    d_target = drift(Mn, Ma, c)
    print(f"\n  phase progression (median deg/day, active consecutive days):")
    print(f"    VPM as stored {d_before:+6.2f}   after transform {d_after:+6.2f}   "
          f"target {d_target:+6.2f}")
    if d_after <= 0:
        raise SystemExit("REFUSING to write: the transformed index runs westward. "
                         "An MJO index must advance eastward through phases 1-8.")
    print("    -> eastward after the transform, as an MJO index must be")

    np.savez(args.out, rotation=R, angle_deg=ang, determinant=det,
             is_reflection=bool(det < 0), target_scale=scale,
             n_days=len(c), span=f"{c[0]}..{c[-1]}",
             vpm_source=args.vpm, rmm_source=args.rmm,
             note="PSL RMM* -- NOT BOM official RMM; re-measure before submitting")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
