# Direction note — explanation first, risk downstream

**Date:** 2026-08-06 · **Status:** adopted; implemented in code as of commit following
`a988c44` · **Supersedes** the implicit scope of `S2S_BN_ONTOLOGY.md` (admin-1 risk as the goal)

---

## The question

> *Are we too focused on risk at admin-1 level? Should we dislodge from that and focus on
> explaining the forecast — the global circulation opportunity that AIFS-ENS v2's full 3-D
> output in the Icechunk store creates — with risk as a prelude rather than the goal?*

## The decision

**Yes. The circulation explanation record becomes the primary product; the Bayesian risk
network becomes a downstream consumer of the same loci.** Demoted, not deleted.

This is not a change of technique — the loci, the registry, the counting rule and the ESS
discipline all survive. It is a change of *what the pipeline is for*, and therefore of what
counts as done.

## Why — the argument from what we actually found

**1. Every weak link is on the risk side; every solid one is on the explanation side.**
Building the admin-1 BN surfaced four blocking dependencies that do not exist yet: no
CHIRPS/IMERG ingest, no EM-DAT, no exposure or vulnerability layer, no elicitation panel. And
three defects that are structural, not fixable by more code: the antecedent node is ~18 days
stale *and* not independent of the forecast (same model run); `P(W|P,A)` and `P(RO|W)` are
elicited by us with nothing able to close them; the moisture node degenerates once pooled to a
polygon. The explanation layer needs none of that. It needs the store, which exists, and ERA5
for verification, which is a well-posed problem in a way impact verification is not.

**2. The scale argument, applied consistently.** Part 0 of `CPT_BUILD_CRITIQUE_AND_PLAN.md`
says predictability at days 18–33 lives at planetary and synoptic scale, and that a locus is
admissible only where its scale retains predictability. Admin-1 pooling takes that
planetary-scale signal, projects it onto polygons where predictability is weakest, and then
multiplies it by elicited numbers nobody has validated. We wrote the rule and then partly broke
it by chasing admin-1 units.

**3. An asymmetry worth naming.** The state-definition bug (registry header; `S2S_BN_ONTOLOGY.md`
§3) was that within-ensemble quantiles are informationless *as risk states*. But *"n of 50
members place the atmosphere in regime B"* is exactly the right form for an explanation product.
**Ensemble-relative framing is native to explanation and fatal to calibrated risk.** The same
property that broke the risk states is the substrate of the explanation product.

**4. It is what the store is uniquely for.** The submission pipeline keeps 3 of 120 variables;
the store keeps the full 3-D state at native N320. The acceptance argument this whole folder
opens with — that MLWP needs *understanding, translating, communicating*, not only skill — is
an argument for the explanation product specifically. Risk framing routes the store's advantage
through a bottleneck that discards it.

## What changes

| | Before | After |
|---|---|---|
| Unit of analysis | admin-1 polygons | circulation regions and modes (`IGAD_EA`, `EQ_INDIAN`, `WIO`, `CONGO`); admin-1 optional |
| Primary output | risk grade per admin-1 | **circulation record**: ensemble counts over circulation states + a generated narrative, ESS attached to every count |
| Verification | impact verification (needs data we lack) | circulation diagnostics vs **ERA5** for the same valid window |
| Customer | anticipatory-action trigger | duty forecaster; and the *"why did the model favour this quintile"* loop back to the AI-WQ submission |
| Risk network | the goal | a consumer, switched on when the observation side exists |

## What this costs — the new first-class blocker

Under risk framing, the derivative problem was deferrable: velocity potential χ200, the
divergent/Walker circulation, moisture-flux convergence and vorticity all need spatial
derivatives, which are undefined on the store's unstructured reduced-Gaussian `values` axis, so
they sat in `blocked_loci` and we moved on.

**Under explanation framing those diagnostics *are* the product.** A regrid step
(`earthkit-regrid` to a regular grid) or a spherical-harmonic transform is now required work,
not a registered absence. The registry records this priority change.

Second cost: the reduced full-window field set (0–792 h, 12-hourly, a handful of fields) moves
from "nice later" to necessary — MJO propagation and moisture-corridor life-cycle cannot be
tracked from hour 432, and those are core explanation content rather than optional extras.

## What is kept

The registry and its provenance discipline (substrate-independent, and the real asset); the
lead-window stratification; the ESS discount; the one-state-per-member joint-draw counting; the
generative direction of `NETWORK_DIRECTION.md`; and the whole BN, unchanged in mechanism.

## Revised work order

**E1 — Circulation record.** *Done in this commit.* Regime occupancy, |IVT| distribution and
corridor-threshold exceedance counts, Somali-jet index, rainfall-environment components,
precipitation-rate distribution, ESS, and a narrative sentence — per region per lead window.
**E1b — Low-level jets by physical core speed.** *Done.* Somali/Findlater and Turkana, with
literature thresholds, a p95-over-box core metric and a nocturnal 0300-local subset
(`low_level_jets.py`). Worth noting for the roadmap: the jets carry **higher ESS (2.7–3.1 and
6.3–7.0) than the regional circulation fields (1.3–1.9)** — members differ more in jet core
time-series than in the broad z500 pattern, so **the jets may be the best-supported explanation
locus this ensemble has**. (Caveat: the two ESS figures are computed over different feature
spaces — jet time-series vs spatial maps — so treat the comparison as indicative.)
**E2 — Regrid or spherical-harmonic step** → χ200, divergent circulation, MFC, vorticity. The
blocker that now matters most.
**E3 — Reduced full-window field set** → MJO propagation, corridor life-cycle from genesis.
**E4 — Model climatology accumulation** → grade-C loci become true anomalies within a season.
**E5 — ERA5 verification** of the circulation diagnostics over the day-18–33 valid window.
**E6 — Quintile lineage** — join the record to the AI-WQ submission and answer *why* the model
favoured a quintile from the ingredients.
**R-track** — the water-balance BN resumes when Phase-2 observations land.

## What would show this decision to be wrong

Stated up front, so the pivot is falsifiable rather than a preference:

1. **Forecasters find counts-without-calibration unusable.** If the duty forecaster wants a
   probability they can act on and not "31 of 50 members, ESS ≈ 2", the explanation product is
   an academic artefact and risk framing was right.
2. **ERA5 verification shows the regime assignment is unstable** — if the same cycle's regimes
   don't correspond to anything verifiable, the narrative is confident noise.
3. **ESS ≈ 1.85 undercuts the explanation product too.** ✅ **TESTED — AND IT FIRED. See below.**

### Falsifier 3 was run, and it went against us

`k_regimes_test.py` (artifact `k_regimes_test_20260730.json`) clusters two independent 80 %
member subsamples per *k* and compares their labels on the intersection with the Adjusted Rand
Index — chance-corrected, so no null model is required. Six cases (3 regions × 2 lead windows):

| k | mean subsample ARI | verdict | silhouette |
|---|---|---|---|
| **2** | **0.817** | **stable** | 0.43 |
| 3 | 0.619 | marginal | 0.36 |
| **4** *(was production)* | **0.557** | **UNSTABLE** | 0.24 |
| 5 | 0.544 | unstable | 0.26 |
| 6 | 0.535 | unstable | 0.24 |

PC1 alone carries **72–88 %** of the ensemble variance in every case (ESS 1.3–1.9 of 50). The
k = 4 partition was manufacturing four regimes out of roughly one dominant direction: re-drawing
the members reshuffles the labels almost as much as it preserves them.

**Action taken — production moved to k = 2.** `loc.circ.v2` supersedes `loc.circ.v1`; counts
never pool across the two ids. Python, the Julia BN (`C_STATES`, `P(M|C)` now 4×2, the model's
`Categorical`/`diageye` arity, the CSV reader and the self-test) and the registry all follow.

**And the honest consequence, exactly as anticipated:** the product is thinner than E1 first
implied. The defensible statement is *"the ensemble carries two circulation stories this
cycle, split n/50–m/50"* — not a four-regime occupancy fan. That is a smaller claim, and it is
the one the data supports. Nothing about the pivot itself is invalidated: the failure was in how
finely we partitioned, not in whether circulation explanation is the right product. But it is a
direct warning about the rest of E1 — **any diagnostic that slices 50 low-DOF members into many
categories should be assumed guilty until stability-tested.**

*(Method note: for `k = 2` the EQ_INDIAN week-4–5 case is the one exception, ARI 0.53 — that
single case prefers k = 3. One case out of six does not overturn the aggregate, but it is the
first place to look if the regional records start disagreeing.)*

### …and the same standard, applied to M / R / P, went against us again

`mrp_stability_test.py` (artifact `mrp_stability_20260730.json`). The k-test's protocol does
**not** transfer, and saying why matters: circulation states are *fitted* to the members by
k-means, so re-drawing members can change every label — that is what the ARI measured. M/R/P
states come from **absolute thresholds**, so a member's state is a deterministic function of its
own value; subsample re-labelling is trivially perfect and would produce a reassuring number
with no content. Tested instead: degeneracy, cut fragility (kernel density *at* the cut ÷ peak
density — the threshold analogue of a silhouette: a good cut sits in a valley, not through a
peak), threshold sensitivity (TVD under ±10 %/±25 % shifts of every cut), and sampling error.

| Node | mean states occupied (of 4) | verdicts over 6 cases |
|---|---|---|
| **M** moisture transport | **1.33** | 4 × DEGENERATE, 2 × FRAGILE |
| **R** rainfall environment | 2.0 | 1 robust, 1 degenerate, 4 fragile |
| **P** precipitation forcing | 2.0 | 1 robust, 1 degenerate, 4 fragile |

Cuts that fall inside the ensemble range land almost exactly **on the density peak**
(density-at-cut 0.99, 0.91, 0.89, 0.85, with 10–11 of 50 members within 5 % of the spread of the
line). At `n_eff = ESS` the 90 % bootstrap CI on a reported state fraction is **0.33–1.00 wide**.

**Decision — thresholds are NOT re-tuned.** Sliding cuts into this cycle's density valleys is
exactly the threshold-re-tuning failure the critique names (C3): it fits the partition to one
event and destroys the cross-cycle comparability that absolute thresholds exist to provide.
Instead:

1. every explanation record now carries a `state_resolution` block per node (occupancy,
   density-at-worst-cut, flag), and the narrative says so in words;
2. the product **leads with continuous ranges and named-threshold exceedance counts**
   (*"|IVT| 158–226, 0 of 50 above 250"*) — a single named physical threshold with a count is
   immune to this failure mode — and demotes the multi-way state split to BN input;
3. the real fix is the **model climatology (E4)**, so cuts sit at climatological percentiles: a
   fixed, non-event-specific reference. **This test is the strongest argument yet for moving E4
   ahead of E2 in the work order.**

The 4-state vocabularies stay as fixed contracts — a degenerate node is itself information
(*"this cycle sits entirely in normal moisture supply"*) — and the ESS discount already keeps
their CPT rows wide.

**Taken together, the two tests say the same thing in two different ways: this ensemble supports
statements about *where the distribution sits relative to named physical thresholds*, and does
not support fine categorical partitions of any kind.** That is a real constraint on the
explanation product, discovered before it was published rather than after.

---

*Implemented by `s2s_bn_evidence_prep.py --explain-out` (primary path) with
`--regions IGAD_EA,EQ_INDIAN,WIO,CONGO`. The BN (`s2s_water_balance_bn.jl`) is unchanged in
mechanism and re-typed as downstream. Registry v0.2.0 carries the new regions, the
`explanation_products` entry, and the blocked-loci priority change.*
