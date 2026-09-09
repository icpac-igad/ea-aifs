# Phase-0/1 implementation — what was built and how it answers the critique

Implements the *first concrete actions* of `CPT_BUILD_CRITIQUE_AND_PLAN.md` against the real
`cycle-20260730_0000` Icechunk store. Everything here is forecast-side (store only); the
observation pipeline (Phase 2, CHIRPS/IMERG) is not yet built.

## Files

| File | Role |
|---|---|
| `NETWORK_DIRECTION.md` | Phase-0 contract: **generative** direction (latent `H`, evidence children `P(E\|H)`), fixing critique **A2**. Cited by the code. |
| `discretization_registry.yaml` | Phase-0 contract: versioned locus/state/lead-window/region IDs. Changing a definition mints a new id; counts never pool across ids (kills failure C2/C3). |
| `cpt_build.py` | The corrected Route-A counter. |
| `cpt_artifact_20260730.nc` | The CPT artifact of record (schema **B7**: raw counts + provenance). |

## How each critique point is enforced in code

| Critique | Where / how |
|---|---|
| **A1** pseudo-replication (50×61) | `build_window`: each member reduced to **one** (regime, IVT) state per lead window; count over the member axis only → `n ≤ 50`. Verified: `counts.sum()==50` per window. |
| **A2** wrong arrow direction | `NETWORK_DIRECTION.md` + artifact `typing` attr: output is within-cycle soft evidence about `H` (`n_cycles=1`), **not** `P(E\|H)`. |
| **A3** `tp` de-accumulation | `verify_tp_interval()` runs first and **asserts**; on the store `tp` is interval (31.7 % cells decrease step-to-step) → no de-accumulation. Grade-A loci use no climatology at all. |
| **A5** derivation-independence | IVT is the **sole** moisture representative; `tcw`/mfc/`tp` excluded (registry `excludes`, artifact attr). Product-of-likelihoods stays valid. |
| **B1** lead stratification | Two AI-WQ-aligned windows (`lw.W1.v1`, `lw.W2.v1`); every table carries exactly one. |
| **B3** ESS + prior | ESS = **participation ratio** of the ensemble-spread eigenspectrum (not mean-pairwise-corr, which is forced ~0 by mean removal). Counts **down-weighted** to ESS (`counts_eff = counts·ESS/n`); symmetric Dirichlet `α₀=max(1,ESS)`; ESS/α₀ recorded. |
| **B7** artifact schema | netCDF: `counts` (raw, always), `counts_eff`, `posterior`, `regime_occupancy`, `ess`, `redundancy`, `alpha0`, + attrs: source tag, registry version, locus IDs, grades, lead-window hours, `tp` semantics, admissibility note. |
| **Part 0** scale admissibility | Every locus is region-pooled (IGAD box) **before** classification; no grid-cell node is ever emitted. |

## The headline result (and why it matters)

On `cycle-20260730_0000`, over the IGAD box (11 342 N320 cells), both lead windows give

```
ESS ≈ 1.8–1.9   (redundancy ≈ 0.96)
```

**The 50-member z500 spread over East Africa at days 18–33 collapses onto ~2 effective spatial
degrees of freedom.** This is itself an *understanding* diagnostic (extended-range ensemble
spread is low-dimensional), and it is exactly why the raw member counts must not be trusted at
face value:

```
raw counts W1 (regime × [ivt_low, ivt_high]):   [[9,6],[2,6],[10,6],[4,7]]
```

The raw counts *suggest* a regime→moisture link (regime 1 is 6/8 moist, regime 0 is 9/15 dry),
but with ESS≈2 the ESS-discounted `P(IVT_high | regime)` posterior is deliberately wide
(0.45–0.60). A single forecast **cannot** establish this relationship — it can only contribute
its raw counts to the Phase-3 pair archive, where the pattern either recurs across cycles
(becoming a real `P(E|H)`) or does not. The raw counts are retained precisely so that later
accumulation, and any better ESS/α policy, can be reapplied without re-running.

That is the whole point of the framework: **the code refuses to turn one low-DOF forecast into
a confident conditional.**

## Run

```bash
python cpt_build.py \
  --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 \
  --tag cycle-20260730_0000 \
  --out cpt_artifact_20260730.nc
```

## Not done (honest scope)

- **Phase 2** observation pipeline (CHIRPS/IMERG ingest, upscaling, R1 antecedent node,
  OISST boundary node) — needs external data; registry ids reserved.
- **Phase 3** matched forecast/obs archive → the real `P(E|H)`; needs many cycles (the
  artifact's `n_cycles=1` makes the gap explicit).
- **Phase 5** null-model skill harness and the L4 elicitation seam.

The build order in the plan is deliberately sequenced so these do not block the Phase-1
machinery, which is what this delivers.
