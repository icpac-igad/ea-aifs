#!/usr/bin/env python3
"""How much do the Cy50r1-only input fields change an AIFS-ENS v2 forecast?

Thirteen of the model's 112 input fields did not exist before IFS Cy50r1 (12 May 2026):
``cdww``, ``wmb``, the six period-band wave heights ``h1012``..``h2530``, and the five
fields of the new 10 hPa level (``t/u/v/w/z_10``).
For any earlier date they would have to be faked. This measures what that costs.

**Design.** aifs-ens-2.0 is stochastic, so a single control-vs-treatment pair would be
swamped by ensemble noise (2t rms ~2.8 K between realisations). But the noise is driven
by torch's global RNG, so seeding pairs the runs: with the same seed two rollouts agree
to ~0.16 K at 24 h (residual FP16 GPU non-determinism) versus ~12.8 K for different
seeds. Every configuration is therefore run under the *same* seeds, and the control run
under a *second* seed gives the ensemble-noise yardstick to judge the result against.

Configurations:
  control  the real fields, as the operational input provides them
  zero     the 8 fields set to 0.0            (a plausible cheap fill)
  nan      the 8 fields set to NaN            (mimics "field absent"; the model already
                                               sees NaN over land in these variables)
  wmb      only ``wmb`` zeroed  -- the runner reports wmb as an unsupported coupled
                                   forcing, so this tests whether it is used at all
  l10copy  10 hPa level carried up from 50 hPa   (reconstruction without a donor)
  l10zero  10 hPa level zeroed                   (pessimistic bound)
  pre50r1  wave fields NaN + 10 hPa from 50 hPa  (best effort with no external data)

Three further modes use *real donor data* for the pre-50r1 fields, fetched by
``fetch_era5_l10.py`` and ``fetch_j1r2.py``. These are the ones that matter: the modes
above only ever measured how bad an invented field is, and ``l10copy`` at 5.5x the
ensemble spread said "do not run". A donor is a different question -- how far a *real*
analysis from another system sits from the operational one.

  l10era5   10 hPa replaced by ERA5 10 hPa      (--era5-l10 donor.npz)
  wavej1r2  8 wave fields replaced by j1r2      (--wave-donor donor.npz)
  pre50r1b  both -- the reconstructed pre-50r1 input, as it would actually be built

Read them against the same floor and ensemble scale as everything else. A donor mode near
the floor means the substitution is invisible to the model; near or above the ensemble
scale means it is not. Note both donors are *deterministic*, so in a real 50-member run
every member would share them -- this measures the mean offset, not the lost spread.

Usage::

    python wave_sensitivity.py --out RESULTS --seeds 0,1 \
        --modes control,zero,nan,wmb,l10copy,l10zero,pre50r1
    python wave_sensitivity.py --out RESULTS --seeds 0,1 \
        --modes control,controlB,l10era5,wavej1r2,pre50r1b \
        --era5-l10 era5_l10_20260813_00.npz --wave-donor j1r2_20260813_00.npz
    python analyse_sensitivity.py --dir RESULTS
"""

import argparse
import os
import sys
import time

os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "16")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

# Cy50r1 added 13 input fields. Before 12 May 2026 NONE of them exist.
WAVE_MISSING = ["cdww", "wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530"]
# ... and a whole new 10 hPa level (q is not carried at 10 hPa).
L10_MISSING = ["t_10", "u_10", "v_10", "w_10", "z_10"]
L10_DONOR = {f: f.replace("_10", "_50") for f in L10_MISSING}   # nearest level below
# Wave fields that DO exist pre-50r1 and are left untouched: swh, mwd (cos/sin), mwp.
CAPTURE = ["2t", "tp", "msl"]


def load_donor(path, expect):
    """Load a donor npz and check it covers ``expect`` on the state's own grid."""
    if path is None:
        return None
    d = np.load(path)
    have = set(d.files)
    missing = [f for f in expect if f not in have]
    if missing:
        raise SystemExit(f"{path}: missing {missing} (has {sorted(have)})")
    return {f: np.asarray(d[f]) for f in have}


def apply_fill(state, mode, era5_l10=None, wave_donor=None):
    """Return a copy of ``state`` with the pre-50r1-missing fields replaced.

    ``l10copy`` is the best reconstruction available with no external data -- carry the
    nearest level below (50 hPa) upward. ``*zero`` variants are the pessimistic bound,
    not something anyone would actually do. The ``*era5*``/``*j1r2*`` modes instead swap
    in a real analysis from another system, which is a different question and the only
    one with a route to a runnable pre-50r1 cycle.
    """
    fields = dict(state["fields"])

    def from_donor(donor, names, what):
        if donor is None:
            raise SystemExit(f"mode needs {what}; pass it on the command line")
        for f in names:
            if f not in fields:
                continue
            src = np.asarray(donor[f])
            dst = np.asarray(fields[f])
            if src.shape != dst.shape:
                raise SystemExit(f"{f}: donor {src.shape} vs state {dst.shape} -- "
                                 f"the donor must be on the state grid (N320)")
            fields[f] = np.array(src, dtype=dst.dtype, copy=True)

    def fill(names, val):
        for f in names:
            if f in fields:
                fields[f] = np.full_like(np.asarray(fields[f]), val)

    def copy_from_donor(names):
        for f in names:
            donor = L10_DONOR[f]
            if f in fields and donor in fields:
                fields[f] = np.array(fields[donor], copy=True)

    if mode in ("control", "controlB"):
        # controlB is byte-identical to control. Run under the SAME seed it measures the
        # irreducible FP16/cuDNN non-determinism floor -- which chaos amplifies over a
        # 33-day rollout, so at day 18-33 it is the only honest baseline to judge a
        # treatment against.
        pass
    elif mode == "zero":
        fill(WAVE_MISSING, 0.0)
    elif mode == "nan":
        fill(WAVE_MISSING, np.nan)
    elif mode == "wmb":
        fill(["wmb"], 0.0)
    elif mode == "l10zero":
        fill(L10_MISSING, 0.0)
    elif mode == "l10copy":
        copy_from_donor(L10_MISSING)
    elif mode == "pre50r1":                 # best effort with no external data
        fill(WAVE_MISSING, np.nan)
        copy_from_donor(L10_MISSING)
    elif mode == "l10era5":                 # real 10 hPa analysis, wrong model cycle
        from_donor(era5_l10, L10_MISSING, "--era5-l10")
    elif mode == "wavej1r2":                # the hindcast the wave inputs were trained on
        from_donor(wave_donor, WAVE_MISSING, "--wave-donor")
    elif mode == "pre50r1b":                # the reconstructed pre-50r1 input
        from_donor(wave_donor, WAVE_MISSING, "--wave-donor")
        from_donor(era5_l10, L10_MISSING, "--era5-l10")
    else:
        raise ValueError(mode)
    out = dict(state)
    out["fields"] = fields
    return out


def main():
    ap = argparse.ArgumentParser(description="pre-50r1 wave-field sensitivity test")
    ap.add_argument("--input-dir",
                    default="/tank/projects/aifs-run/20260813_0000/input_states")
    ap.add_argument("--member", type=int, default=1)
    ap.add_argument("--lead-time", type=int, default=792)
    ap.add_argument("--write-hours", default="432-792",
                    help="capture only steps in this window (the product window)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--modes", default="control,zero,nan,wmb")
    ap.add_argument("--era5-l10", default=None,
                    help="npz from fetch_era5_l10.py (needed by l10era5/pre50r1b)")
    ap.add_argument("--wave-donor", default=None,
                    help="npz from fetch_j1r2.py (needed by wavej1r2/pre50r1b)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import fp16_multi_run_AIFS_ENS_v2 as runmod
    from anemoi.inference.runners.simple import SimpleRunner

    os.makedirs(args.out, exist_ok=True)
    lo, hi = (int(x) for x in args.write_hours.split("-"))
    seeds = [int(s) for s in args.seeds.split(",")]
    modes = args.modes.split(",")

    # Fail on a bad donor now, not 40 minutes into the first rollout.
    era5_l10 = load_donor(args.era5_l10, L10_MISSING)
    wave_donor = load_donor(args.wave_donor, WAVE_MISSING)
    for m, need, flag in (("l10era5", era5_l10, "--era5-l10"),
                          ("wavej1r2", wave_donor, "--wave-donor"),
                          ("pre50r1b", era5_l10, "--era5-l10"),
                          ("pre50r1b", wave_donor, "--wave-donor")):
        if m in modes and need is None:
            sys.exit(f"mode {m} needs {flag}")

    runner = SimpleRunner({"huggingface": "ecmwf/aifs-ens-2.0"}, device="cuda",
                          precision="16")
    base = runmod.load_input_state_from_pickle(args.member, args.input_dir)
    runner.time_step = 6
    runner.lead_time = args.lead_time
    runner.reference_date = base["date"]
    all_cy50r1 = WAVE_MISSING + L10_MISSING
    absent = [f for f in all_cy50r1 if f not in base["fields"]]
    print(f"Cy50r1-only fields present in this input: "
          f"{len(all_cy50r1) - len(absent)}/{len(all_cy50r1)}"
          + (f" (absent: {absent})" if absent else ""))

    for seed in seeds:
        for mode in modes:
            tag = f"{mode}_seed{seed}"
            path = os.path.join(args.out, f"{tag}.npz")
            if os.path.exists(path):
                print(f"[skip] {tag} already done")
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            state = apply_fill(base, mode, era5_l10, wave_donor)
            caught = {v: [] for v in CAPTURE}
            hours = []
            for step, s in enumerate(runner.run(input_state=state,
                                                lead_time=args.lead_time), start=1):
                h = step * 6
                if lo <= h <= hi:
                    hours.append(h)
                    for v in CAPTURE:
                        caught[v].append(np.asarray(s["fields"][v], dtype="float32"))
            np.savez_compressed(path, hours=np.array(hours),
                                **{v: np.stack(caught[v]) for v in CAPTURE})
            print(f"[ok] {tag}: {len(hours)} steps in {time.time() - t0:.0f}s "
                  f"-> {os.path.getsize(path) / 1e6:.0f} MB")

    print("done")


if __name__ == "__main__":
    main()
