"""Does a 144-longitude band series propagate eastward like the MJO?

This is the diagnostic `MJO_PHASE.md` 6b needed and did not have. 6b measured
the drift of the wavenumber-1 crest in *raw* chi200, found it stationary, and
concluded the MJO was buried under the Walker cell. The conclusion was right but
the test could not support it, for two reasons this module fixes.

**1. Crest-phase drift is unusable when the wave is weak.** It needs phase
unwrapping, and once the wavenumber-1 amplitude drops the phase is noise, so the
fitted slope swings wildly (measured: -18, +3, +12 deg/day on the three cycles
after removing a stationary field). `phase_speed` uses lag-longitude
correlation instead, which never unwraps anything: a weak signal degrades it to
a low correlation rather than to a confident wrong answer.

**2. A 34-day window cannot define its own MJO anomaly.** The MJO period is
30-60 days, so subtracting the mean of a 34-day forecast window removes a large
part of the signal along with the Walker cell. That is why 6b's proposed test -
"remove a climatology, recompute the drift" - cannot decide anything on its own:
an absent MJO and a broken pipeline both leave an incoherent residual. A
multi-month ERA5 window *is* long enough to self-reference, which is what makes
`era5_vpm_clim.py --mode series` the decisive control.

`bandpass` is the classic 20-100 day MJO filter. It needs a window several times
the longest period to be meaningful; `min_days` refuses rather than silently
returning a filtered-looking series that is mostly edge effect.
"""
from __future__ import annotations

import numpy as np

DX = 360.0 / 144        # deg longitude per bin
MJO_SPEED = (4.0, 8.0)  # deg/day eastward, the phase speed to be recovered


def bandpass(series, lo=20.0, hi=100.0, min_days=None):
    """Retain periods in [lo, hi] days. `series` is (ndays, nlon), daily."""
    n = series.shape[0]
    need = min_days if min_days is not None else 2.0 * hi
    if n < need:
        raise ValueError(f"{n} days is too short for a {lo}-{hi} day bandpass "
                         f"(need >= {need:.0f}). A window shorter than the longest "
                         f"retained period cannot separate the band from its own mean.")
    f = np.fft.rfftfreq(n, d=1.0)                  # cycles/day
    keep = (f >= 1.0 / hi) & (f <= 1.0 / lo)
    F = np.fft.rfft(series - series.mean(0, keepdims=True), axis=0)
    F[~keep] = 0.0
    return np.fft.irfft(F, n=n, axis=0)


def phase_speed(A, maxlag=12, min_corr=0.2):
    """(deg/day, diagnostics) from a (ndays, nlon) anomaly. Positive = eastward.

    For each lag the zonal displacement that maximises the pattern correlation is
    found by cyclic shift; the speed is a correlation-weighted fit through the
    origin. Lags whose best correlation falls below `min_corr` carry no weight,
    so a decorrelated field reports a small weight rather than a large speed.
    """
    A = A - A.mean(axis=0, keepdims=True)
    A = A - A.mean(axis=1, keepdims=True)           # drop the zonal mean each day
    n = A.shape[1]
    s = np.arange(-(n // 2), n // 2)
    lags, disp, peak = [], [], []
    for lag in range(1, maxlag + 1):
        a, b = A[:-lag], A[lag:]
        c = np.array([np.corrcoef(a.ravel(), np.roll(b, k, axis=1).ravel())[0, 1]
                      for k in s])
        k = int(np.argmax(c))
        # roll(b, k) moves the t+lag pattern k bins east to match t, so the
        # feature itself moved -k bins east between t and t+lag.
        lags.append(lag); disp.append(-s[k] * DX); peak.append(c[k])
    lags, disp, peak = np.array(lags, float), np.array(disp), np.array(peak)
    w = np.where(peak >= min_corr, np.clip(peak, 0, None) ** 2, 0.0)
    if w.sum() <= 0:
        return np.nan, dict(lags=lags, disp=disp, corr=peak, weight=w)
    speed = (w * lags * disp).sum() / (w * lags * lags).sum()
    return speed, dict(lags=lags, disp=disp, corr=peak, weight=w)


def verdict(speed, diag, lo=MJO_SPEED[0], hi=MJO_SPEED[1]):
    if not np.isfinite(speed):
        return "NO COHERENT PROPAGATION (every lag below the correlation floor)"
    c1 = diag["corr"][0]
    tag = "EASTWARD, MJO-like" if lo <= speed <= hi else (
          "eastward but outside 4-8" if speed > 0 else "WESTWARD")
    return f"{tag} | lag-1 corr {c1:.2f}"
