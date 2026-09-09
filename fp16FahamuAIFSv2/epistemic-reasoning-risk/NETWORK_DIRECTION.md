# Network direction — Phase 0 contract (fixes critique A2)

**Version 0.1.0.** Cited by every CPT/BN script. Changing this changes the meaning of every
artifact; bump the version.

## The generative structure (not the discriminative one the drafts used)

The earlier drafts wired evidence *into* the outcome as parents
(`P(outcome | forecast × station)`) — which needs an unfillable joint table and cannot fuse
independent evidence (critique A2). We invert it.

```
                 ┌──────────────────────────────┐
                 │   H  (latent)                 │
                 │   hazard-relevant circulation │
                 │   / rainfall state, per       │
                 │   admin-1 unit, per lead      │
                 │   window                      │
                 └───────────────┬──────────────┘
        ┌───────────────┬────────┼─────────┬───────────────────┐
        ▼               ▼        ▼         ▼                   ▼
  P(z500 regime|H) P(IVT state|H) P(Somali|H) P(CHIRPS antecedent|H) P(station|H)
   (forecast)      (forecast)    (forecast)   (observation, Phase 2)  (obs, Phase 2)
```

- **H is latent.** Evidence nodes hang *off* H as children. Each likelihood `P(Eᵢ | H)` is
  learned independently from matched pairs (forecast-state ↔ observed-outcome for the
  forecast children; obs-product ↔ outcome for the observation children).
- **Posterior:** `P(H | E) ∝ P(H) · Πᵢ P(Eᵢ | H)` — a product of independently-calibrated
  likelihoods, which is coherent; the discriminative version was not.
- **Derivation-independence audit (A5):** the product-of-likelihoods form assumes the `Eᵢ`
  are conditionally independent given H. Diagnostics sharing source fields (IVT, `tcw`,
  moisture-flux-convergence, `tp` all come from `q,u,v`) are **not** independent — admit at
  most **one representative per correlated cluster** (we take IVT for the moisture role). The
  registry's `derivation_fields`/`excludes` encode this.

## What Phase 1 actually produces (honest typing)

Phase 1 has **one cycle**. It cannot estimate `P(Eᵢ | H)` (that needs matched forecast/obs
pairs across many cycles — Phase 3). Phase 1 produces the **within-cycle co-occurrence
structure** of the forecast loci — "given this forecast's uncertainty, how do the moisture
state and the synoptic regime co-vary across the 50 members" — carrying `n_cycles = 1` in the
artifact. It enters the network as **soft evidence about H for this cycle**, never as the
likelihood calibration. Conflating the two is the overconfidence the framework exists to
prevent.

## Counting rules baked into the code (fix A1, B3)

1. **One state per member per lead window** before counting → `n ≤ 50` (not 50×61 = 3050).
   Consecutive 6-h steps are autocorrelated; the flattened count is pseudo-replication.
2. **Effective sample size** `ESS = n / (1 + (n−1)·ρ̄)`, `ρ̄` = mean pairwise spatial
   correlation of the members' regional anomalies. Recorded in every artifact.
3. **Dirichlet prior** with total strength `α₀ = max(2, ESS)` (prior as strong as the
   effective data) → early posteriors are wide *by construction*. `α₀` and its basis recorded.
4. **Raw count tensor is always stored** (never only the normalized table), so a later, better
   prior can be reapplied without re-running inference.

## Scale × lead admissibility (Part 0)

A locus is admitted at a lead only if its scale retains predictability there. At days 18–33:
planetary/boundary modes → parents; synoptic (regimes, IVT) → member-counted loci **per lead
window**; meso/grid-wise → **never** direct evidence, only pooled or as an obs-learned
`P(admin-1 outcome | H)` row. The code refuses to emit a grid-cell evidence node.
