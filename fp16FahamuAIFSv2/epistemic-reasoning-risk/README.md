# Epistemic reasoning & risk from the AIFS-ENS Icechunk store

Turning a per-cycle AIFS-ENS-2.0 forecast into an **auditable, calibrated belief** about
hazard-relevant weather over East Africa at the S2S range (days 18–33) — not a skill score.

The premise (from an MLWP-acceptance argument): the operational value of ML weather prediction
cannot be judged by verification skill alone; acceptance depends on **understanding, translating
and communicating**. The per-cycle Icechunk store (120 variables, 50 members, native N320) is
the substrate that makes those functions possible — the `tp/msl/2t` quintile submission
discards everything else. This folder builds the reasoning layer on top of it.

## The arc (why the files exist, in order)

1. **`../ICECHUNK_S2S_DIAGNOSTICS.md`** — *understand.* What the store lets a forecaster
   diagnose that the submission throws away: IVT/atmospheric rivers, ENSO/IOD fingerprints,
   MJO, Somali jet, weather regimes. Grounded in the verified variable set.
2. **`epistemic-reasoning-summary.md`** + **`main.tex`** — *the framework.* Evidence grades
   (A–D, hard/soft/virtual), routes for building conditional probability tables (CPTs) from the
   store and observations, and the risk-communication seam (CRMA grades).
3. **`CPT_BUILD_CRITIQUE_AND_PLAN.md`** — *the critical turn.* A hard critique of (1)+(2) —
   pseudo-replication, backwards network arrows, unverified `tp`, mis-typed SST nodes,
   over-sold convergence — and a corrected, phased build plan (Part 0 scale×lead, Phases 0–5).
4. **Phase-0/1 implementation** (below) — *the corrected code*, run on a real cycle.
5. **`EVIDENCE_MAP_AND_BASIN_PLAN.md`** — *the evidence map.* The six evidence nodes resolved
   against verified store contents, plus the HydroBASINS level-3 domain plan. Note §4 is
   partly superseded by (6).
6. **`SHORT_RANGE_EVIDENCE_MAP.md`** — *the same six nodes at 0–7 days*, with exact fields
   rather than concepts. Records that the ∇·/∇× blocker in (5) §4 was already solved in
   `ts-mjo/grid_ops.py` five days before it was written down, re-verifies the operator on O96,
   and argues that CAPE — the one real gap — is better served by stability indices than by
   MU-CAPE at this vertical resolution.

## Phase-0 contracts (fix the failure modes before any counting)

| File | Fixes | Content |
|---|---|---|
| `NETWORK_DIRECTION.md` | **A2** | Generative structure: latent hazard state `H`, evidence as **children** `P(E\|H)`, posterior ∝ prior · ∏ likelihoods. Not the discriminative `P(outcome\|forecast×station)` that needs an unfillable joint. |
| `discretization_registry.yaml` | **C2/C3** | Versioned locus / state / lead-window / region IDs. Change a definition → new id; counts never pool across ids (kills state-definition drift and threshold re-tuning). |

## Phase-1 implementation

- **`cpt_build.py`** — the corrected Route-A counter. Every critique point enforced in code:
  **A1** one modal state per member per lead window → `n ≤ 50` (not 50×61);
  **A3** `verify_tp_interval()` asserts before any rainfall locus;
  **A5** IVT is the sole moisture representative (no `tcw`/mfc/`tp` double-count);
  **B1** two AI-WQ-aligned lead windows; **B3** ESS = participation ratio of the ensemble-spread
  spectrum, counts down-weighted to ESS, raw counts always kept; **Part 0** region-pooled before
  classification, no grid-cell node.
- **`cpt_artifact_20260730.nc`** — the artifact of record (**B7** schema): raw `counts`,
  `counts_eff`, `posterior`, `regime_occupancy`, `ess`, `alpha0`, and full provenance
  (source tag, registry version, locus IDs, grades, lead hours, `tp` semantics).
- **`IMPLEMENTATION.md`** — how each critique item is discharged, with the run command.

```bash
python cpt_build.py \
  --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 \
  --tag cycle-20260730_0000 --out cpt_artifact_20260730.nc
```

## Initial results (cycle `cycle-20260730_0000`, IGAD box, 11 342 N320 cells)

- **`tp` is interval, not accumulated** (31.7 % of cells decrease step-to-step) → no
  de-accumulation; the critique's A3 gate is closed for this store.
- **Ensemble spread is low-dimensional:** ESS ≈ **1.8–1.9** (redundancy ≈ 0.96) in *both* lead
  windows — the 50-member `z500` spread over East Africa at days 18–33 collapses onto **~2
  effective spatial degrees of freedom**. This is itself an *understanding* diagnostic.
- **A regime→moisture hint that the framework correctly refuses to trust yet.** Raw counts
  (regime × [ivt_low, ivt_high]) for week-3 = `[[9,6],[2,6],[10,6],[4,7]]` — regime 1 is 6/8
  moist, regime 0 is 9/15 dry. But with ESS≈2, the ESS-discounted `P(IVT_high | regime)` stays
  wide (**0.45–0.60**): one low-DOF forecast cannot establish the conditional. The raw counts
  are retained so the pattern can be tested against future cycles (Phase 3).

That refusal is the whole point: **the code will not turn a single low-DOF forecast into a
confident belief at a scale where belief is not yet justified** — the acceptance criterion the
opening argument actually proposes.

## The S2S process BN, and the turn to explanation-first (2026-08-05/06)

5. **`qd-1.md.txt`** → **`S2S_BN_ONTOLOGY.md`** — *the process ontology.* A grouped physical DAG
   (circulation → moisture transport → rainfall environment → precipitation forcing +
   antecedent wetness → catchment water-balance pressure) reconciled against the **verified**
   store contents. Adapts `icpac-igad/bn-ibf@jua-bnet`: `s2s_bn_evidence_prep.py` (store →
   evidence, after `flood_data_prep.py`) and `s2s_water_balance_bn.jl` (RxInfer BN, after
   `flood_bn_ibf_v1.jl`). Key findings: `cape`/`tcwv`/`lsm`/orography **absent**; `ro`/`swvl`
   **present** (runoff enters as an evidence *child*, never the target); `tp`/`cp`/`ro`/`sf`
   are **interval per-6h**; the target is a water-balance *tendency* because the store starts
   at day 18.
6. **`DIRECTION_EXPLANATION_FIRST.md`** — *the turn.* The primary product becomes the
   **circulation explanation record** (ensemble counts over circulation states per region per
   lead window, ESS attached to every count); the risk network becomes a downstream consumer.
   Reason: every blocking dependency and structural defect sits on the risk side, while the
   explanation side needs only the store and ERA5 — and *"n of 50 members"* framing is native
   to explanation and fatal to calibrated risk. Cost: the derivative diagnostics (χ200, MFC,
   vorticity) stop being deferrable and now need a regrid or spherical-harmonic step.
7. **`HANDOVER.md`** — *for a second opinion.* Decisions with the case against each, every
   judgement number in one table, and what is tested vs unrun.
8. **`k_regimes_test.py`** — *the falsifier, run.* The handover's own sharpest self-criticism,
   tested and upheld against us: with ESS ≈ 1.3–1.9 and PC1 holding **72–88 %** of the ensemble
   variance, `k = 4` circulation clustering is **unstable** (subsample-ARI **0.557**) while
   `k = 2` is **stable** (**0.817**). Production moved to k = 2 (`loc.circ.v2` supersedes
   `loc.circ.v1`; counts never pool across them) in Python, in the Julia BN's node arity, and in
   the registry. The defensible statement is *"the ensemble carries two circulation stories,
   split 26/24"* — not a four-regime fan. Artifact: `k_regimes_test_20260730.json`.
9. **`mrp_stability_test.py`** — *the same standard, applied to the threshold nodes.* The ARI
   protocol does **not** transfer (threshold states are a deterministic function of each
   member's own value, so subsample re-labelling is trivially perfect); tested instead for
   degeneracy, cut fragility, threshold sensitivity and sampling error. **The 4-state
   partitions of M/R/P are not resolved either:** M occupies **1.33 of 4** states on average
   (4 of 6 cases fully degenerate), R and P **2.0 of 4**, and cuts inside the ensemble range
   sit almost on the distribution's density peak (up to **0.99**). Thresholds were deliberately
   **not** re-tuned — that is the C3 failure — so records now carry a `state_resolution` flag,
   the product leads with ranges and named-threshold exceedance counts, and the model
   climatology (E4) becomes the priority. Artifact: `mrp_stability_20260730.json`.

10. **`low_level_jets.py`** — *the jets, done properly.* The earlier Somali index was weak and
   the Turkana jet was wrongly written off. Both are now identified by **physical core speed**:
   core = 95th percentile of |V| over the jet box (a percentile, so the core survives pooling),
   with literature thresholds — Somali/Findlater **15 / 25 m/s at 850 hPa** (~1.5 km),
   Turkana **12 / 16.8 / 30 m/s at 925 hPa** (~0.75 km) — plus a direction check and a
   **nocturnal 0300-local** subset (the cycle is 00Z-initialised and East Africa is UTC+3, so
   steps at lead hours divisible by 24 land exactly on the campaign hour). Result on
   `cycle-20260730_0000`: Findlater core 14.7–21.5 m/s with **50/50 members above 15 m/s for
   91 % of week-3 steps**, weakening to 63 % duration by weeks 4–5; Turkana core 10.5–13.2 m/s,
   **50/50 above 12 m/s**, only **5/50** reaching the 16.8 m/s campaign core — and the model
   reproduces the **nocturnal maximum** (core 14.0 m/s at 0300 local vs ~11.8 all-hours).
   Notably the jets carry **higher ESS (2.7–3.1 Somali, 6.3–7.0 Turkana)** than the regional
   circulation fields (1.3–1.9), so they may be the best-supported explanation locus available.
   Artifact: `low_level_jets_20260730.json`.

11. **`upper_jet_and_highs.py`** + **`CROSS_CYCLE_FINDINGS.md`** — *does the recipe generalise,
   and does anything replicate?* The jet recipe applied to the **Tropical Easterly Jet**
   (`u_200`, p95 of the easterly component, 15/25/35 m/s) and the **Mascarene / St Helena
   highs** (`msl`), then everything re-run on a **second cycle** (`20260723`).
   **Replicates:** ESS 1.25–1.87 and PC1 75–89 %; k=4 unstable (ARI 0.565 vs 0.557); jets carry
   2–4× the field ESS; the Somali jet's week-3→4/5 decay; and the Turkana nocturnal core at
   **14.01 vs 14.04 m/s** on two independent cycles. **Weaker:** k=2 is *marginal* (0.688) on
   the second cycle, not "stable" as the first suggested. **TEJ generalises** — thresholds
   discriminate (16/50–50/50), and the Indian sector beats the African sector in all four cases,
   which is the right physics and was never encoded. **Highs: strength saturates** (metric/
   threshold mismatch, left un-retuned) **but ridge position is informative** — the ensemble
   agrees on ridge latitude (sd 1.0–1.5°) and disagrees on longitude (sd 5.9–8.9°).

**What the two tests jointly establish:** this ensemble supports statements about *where the
distribution sits relative to named physical thresholds*, and does **not** support fine
categorical partitions of any kind — clustered or threshold-defined.

```bash
python s2s_bn_evidence_prep.py \
  --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 --tag cycle-20260730_0000 \
  --regions IGAD_EA,EQ_INDIAN,CONGO --explain-out explain_20260730.json
```

## Not built (honest scope)

Forecast-side only. Reserved in the registry, needing external data or many cycles:
**Phase 2** observation pipeline (CHIRPS/IMERG antecedent node, OISST boundary node),
**Phase 3** matched forecast/obs archive → the real `P(E|H)` (the artifact's `n_cycles=1`
makes this gap explicit), **Phase 5** null-model skill harness + the L4 elicitation seam.

---

*`*.txt` are exported session transcripts (run archives). Grounded in the verified 20260730
store: 120 vars, 50 members, N320, hours 432–792.*
