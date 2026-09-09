#!/usr/bin/env python3
"""`tercile_probs` must partition exactly as AI-WQ's scorer partitions the observation.

`AI_WQ_package.forecast_evaluation.conditional_obs_probs` decides which tercile the
*observation* falls in. A forecast binned differently is graded against a different
partition, so this is a conformance test against the installed package, not a unit test
of our own convention. It fails if AI-WQ changes its edges.

    $PY test_tercile_binning.py
"""
import numpy as np
import xarray as xr

from AI_WQ_package.forecast_evaluation import conditional_obs_probs
from ts_days import tercile_probs


def aiwq_category(x, lower, upper):
    """Category AI-WQ's scorer assigns to a single observation x."""
    bounds = xr.DataArray([lower, upper], dims=["tercile"],
                          coords={"tercile": np.arange(1, 3) / 3})
    crit = conditional_obs_probs(xr.DataArray(float(x)), bounds).values
    if np.isnan(crit).any():          # all_equal -> unscoreable
        return None
    hits = np.flatnonzero(crit.astype(bool))
    return int(hits[0]) if hits.size else None


def check(counts, lower, upper):
    probs, degenerate = tercile_probs(counts, lower, upper)

    expect = np.zeros(3)
    unscoreable = 0
    for x in counts:
        cat = aiwq_category(x, lower, upper)
        if cat is None:
            unscoreable += 1
        else:
            expect[cat] += 1
    expect /= len(counts)

    if degenerate:
        # AI-WQ masks the observation entirely; the middle category must stay empty.
        assert probs[1] == 0.0, f"degenerate bounds must emit no 'near': {probs}"
        assert unscoreable == len(counts), "expected AI-WQ to mask every observation"
        return "degenerate"

    assert np.allclose(probs, expect), (
        f"bounds ({lower}, {upper}): ours {probs} != AI-WQ {expect}")
    assert np.isclose(probs.sum(), 1.0), f"probabilities must sum to 1: {probs}"
    return "ok"


CASES = [
    # (bounds, description) -- counts always span both bounds, so ties are exercised
    ((5.0, 8.0),  "20260914 NWP -- the cell where the edges moved 0.14"),
    ((4.0, 7.0),  "20260907 NWP"),
    ((6.0, 11.0), "20260907 ATL -- no member reaches the upper bound"),
    ((5.0, 12.0), "20260914 ATL"),
    ((0.0, 0.0),  "SWIO/SEIO in September -- degenerate, AI-WQ masks it"),
    ((0.0, 3.0),  "lower bound at zero, every count ties or exceeds it"),
    ((2.0, 3.0),  "adjacent integer bounds -- 'near' is a single value"),
    ((1.5, 4.5),  "non-integer bounds -- no ties possible"),
]

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    failures = 0
    for (lo, hi), why in CASES:
        # exhaustive small range + a random draw, both spanning the bounds
        for counts in (np.arange(0, max(hi + 3, 6)),
                       rng.integers(0, int(max(hi + 3, 6)), size=50)):
            try:
                verdict = check(np.asarray(counts, float), lo, hi)
            except AssertionError as exc:
                print(f"FAIL  ({lo}, {hi})  {why}\n      {exc}")
                failures += 1
                break
        else:
            print(f"pass  ({lo:>4}, {hi:>4})  {verdict:10s}  {why}")

    print("\nPASS -- binning matches AI_WQ_package.forecast_evaluation.conditional_obs_probs"
          if not failures else f"\nFAILED {failures}/{len(CASES)}")
    raise SystemExit(1 if failures else 0)
