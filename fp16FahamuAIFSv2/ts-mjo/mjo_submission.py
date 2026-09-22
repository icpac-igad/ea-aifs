"""Reduce the daily MJO product to the (9, 4) array AI-WQ expects.

AI-WQ's MJO submission is **not** the weekly-window shape the gridded variables
use. From `AI_WQ_package.forecast_submission`:

    forecast_lags = np.array((7, 14, 21, 28))
    da = xr.DataArray(..., dims=['MJO_phase','valid_time'],
                      coords=dict(valid_time=fc_issue_time + forecast_lags, ...))

so it is **9 phases x 4 valid times, at init + 7/14/21/28 days** -- four single
days, not four weekly means -- and unlike `tas`/`mslp`/`pr` it is **not split by
forecast period**: one file covers all four lags. `check_fc_submission` enforces
only the `(9, 4)` shape and that each column sums to 1.

`vpm_index.py` writes 34 daily entries indexed from the init date, so the four
columns are day indices 7, 14, 21 and 28 directly.

*** THIS DOES NOT MAKE THE PRODUCT SUBMITTABLE ***
`MJO_PHASE.md` 6g: forecast amplitudes run ~2x observed and P(inactive) is
0.01-0.07 against an observed 0.37, because an *observed* climatology is being
removed from a *model* field. This script only reshapes; it fixes nothing.
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

LAGS = (7, 14, 21, 28)


def to_submission(path, lags=LAGS):
    ds = xr.open_dataset(path)
    P = ds.MJO_phase_probability.values            # (day, 9)
    days = [str(d) for d in ds.day.values]
    if max(lags) >= P.shape[0]:
        raise SystemExit(f"product has {P.shape[0]} days; needs > {max(lags)}")
    out = P[list(lags), :].T                        # -> (9, 4)
    if not np.allclose(out.sum(axis=0), 1.0, atol=1e-6):
        raise SystemExit(f"columns do not sum to 1: {out.sum(axis=0)}")
    return out, [days[i] for i in lags], ds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("products", nargs="+", help="mjo_probs_<date>.nc files")
    ap.add_argument("--out-dir", help="write <init>_MJO.npy alongside, for the submission step")
    args = ap.parse_args()

    for p in args.products:
        arr, dates, ds = to_submission(p)
        init = ds.attrs.get("cycle_init", "?")
        print(f"\n=== {init} === valid {', '.join(d[:10] for d in dates)}  (init +7/14/21/28)")
        print("  phase " + "".join(f"{d[5:10]:>9s}" for d in dates))
        for k in range(9):
            lbl = "inactive" if k == 0 else f"phase {k}"
            print(f"  {lbl:8s} " + "".join(f"{v:9.2f}" for v in arr[k]))
        print("  modal    " + "".join(f"{int(np.argmax(arr[:, j])):9d}" for j in range(4)))
        if args.out_dir:
            f = f"{args.out_dir}/{init}_MJO.npy"
            np.save(f, arr)
            print(f"  wrote {f}  shape {arr.shape}")


if __name__ == "__main__":
    main()
