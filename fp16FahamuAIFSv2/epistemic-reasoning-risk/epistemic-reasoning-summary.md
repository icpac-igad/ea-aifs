# Epistemic Reasoning for Risk Assessment

*Conditional probability tables as the probability attached to each locus of reasoning*

---

## The gap

Impact-Based Forecasting answers an **ontological** question — *what is the risk?* — by
combining hazard, exposure and vulnerability into an estimate of loss. It cannot answer the
**epistemic** one: *given all evidence available now, what should we believe about that risk,
and how justified is the belief?* Triggers and thresholds discard exactly this — confidence,
conflicting observations, and the record of what was known when.

## The epistemic layer

Continuous Risk Monitoring & Assessment (CRMA) supplies that missing layer, structured as a
**Bayesian network** over the IBF ontology. Reasoning is decomposed into **loci** — evidence
nodes (rainfall, ensemble forecast, field report), a latent risk grade, and the decision node.
Each locus carries a **conditional probability table** that states, as odds rather than a hard
rule, how much its parents may move it: *"if soils are wet **and** the forecast indicates
deficit, then…"*.

## Evidence typing governs the CPT

Evidence is typed before it enters, and the type governs which CPT column it activates:

| Type | In a word | Enters the assessment as |
| --- | --- | --- |
| **Hard** — gauges, satellites, river levels | what we **measure** | a near-certain value |
| **Soft** — forecasts, expert judgment, community reports | what we **estimate** | a probability spread (*the spread **is** the confidence*) |
| **Virtual** — storylines, downward counterfactuals | what we **imagine, to prepare** | a reliability-weighted likelihood nudge |

## The outcome

Bayesian updating then revises posteriors coherently each cycle, leaving an inspectable audit
trail: **what we believed, when, and on what evidence**. The output is a risk *indication* —
not a calibrated probability of disaster, and not a decision.

---

# Part II — One evidence source: AIFS-ENS v2 on the Icechunk store

> *"The operational value of machine-learning weather prediction cannot be assessed through
> verification skill alone. Acceptance depends on support for the core functions of operational
> meteorology: **understanding, translating and communicating**. Current MLWP systems show
> strong performance but lack the interpretability, diagnostic depth, and stability required
> for trust in safety-critical contexts."*
> — framing of `ICECHUNK_S2S_DIAGNOSTICS.md`

That paragraph is an **epistemic** claim, not an ontological one. It does not say the model is
wrong; it says we have no basis on which to *justify belief* in it, because a skill score
carries no reasoning. Part I's answer to that is the Bayesian network. Part II asks what the
network looks like when the **only forecast** evidence admitted is a single AIFS-ENS v2 cycle,
held in the per-cycle Icechunk store (`…/aifs-run/<date>_0000/icechunk_v2`, tag
`cycle-<date>_0000`: 120 variables, 50 members, native N320, hours 432–792) — and then, in §9,
what past observations add back.

### How the 50 members are actually produced — and why it matters epistemically

Each member is run from its **own initial condition**: `ecmwf_opendata_pkl_input_aifsens_v2.py`
builds 50 `input_state_member_NNN.pkl` files, member *m* pulling IFS ensemble perturbation
`number=m` from the ECMWF open-data `enfo` stream (atmosphere) and `waef` (waves), over the two
timesteps `[date−6 h, date]`, regridded to N320. AIFS-ENS then propagates each forward.

Three consequences for the network:

1. **The 50 members are 50 distinct initial conditions, not one.** Counts across the member
   dimension therefore *are* a genuine sample of initial-condition uncertainty — CPTs built from
   them (Route A) are better founded than a shared-IC ensemble would allow.
2. **But the perturbation is inherited from the IFS ENS, not native to AIFS.** The spread the
   store contains is IFS initial-condition spread propagated through a different model. There is
   no AIFS-side stochastic-physics or model-error perturbation, so the ensemble is likely
   **under-dispersive with respect to model error**. Every CPT derived from member counts should
   carry that as a stated limitation, not an assumed calibration.
3. **One cycle is one atmospheric situation.** Members sample uncertainty *about this flow*;
   they do not sample the climate. That is what makes Route B (multi-cycle accumulation)
   necessary, not optional.

## 5. Re-arranging the loci around a single forecast

With one forecast source, the hard/soft/virtual typology of Part I **collapses on the forecast
side** — every field is soft, from the same 50-member draw. The epistemic structure has to be
rebuilt on a different axis. It comes from the fact that the forecast is not one number: the
store retains the full state, so **one forecast supports a chain of loci**, and each link in
that chain is a place where belief can be stated, questioned and audited.

| Locus | What it is | Held as | Where it comes from |
| --- | --- | --- | --- |
| **L0 — State** | the raw stored fields | `(member, time, values)` at N320 | the store itself; no inference |
| **L1 — Ingredient** | IVT, moisture-flux convergence, χ200, jet indices, static stability | derived field per member | vertical integrals / Helmholtz split over L0 |
| **L2 — Object / mode** | AR object, weather regime, MJO phase, IOD wind index, Somali-jet state | discrete state per member | detection, clustering, projection over L1 |
| **L3 — Hazard indication** | "wet / dry signal over admin-1 *x*, days 18–33" | probability over states | CPT, parents = L2 |
| **L4 — CRMA grade** | Monitor / Evaluate / Assess / Actionable | colour per admin-1 | CPT, parents = L3 + exposure & vulnerability (IBF) |

**The 50 members are what makes each locus probabilistic.** A locus is not a number carried
forward; it is a *count over members* — `n of 50 members place the atmosphere in regime B`.
That count is the conditional probability, read directly off the store.

### Reliability now scales with derivation distance

The typology that replaces hard/soft/virtual, for a single-source network:

| Grade | Condition | Examples | Weight in the CPT |
| --- | --- | --- | --- |
| **A — Direct** | field is in the store; no proxy, no baseline | `swh`, `msl`, `z500` regimes, IVT (`q,u,v` on 13 shared levels) | full |
| **B — Proxy** | a stand-in is used for the physical quantity | `skt` → SST, `cp`/`strd`/`tcw` → OLR | discounted; the proxy must be named at the node |
| **C — Relative** | needs an anomaly baseline we do not yet hold | IOD wind index, χ200 anomaly, MJO amplitude | *within-ensemble only*; must not be stated as an absolute index |
| **D — Truncated** | needs the discarded 0–432 h window | MJO propagation, AR tracking from genesis | flagged as unavailable, not silently approximated |

This is the honest version of "evidence quality" when there is only one witness: you cannot
grade the witness by corroboration, so you grade the **inference path** instead. A node at
grade C or D is not excluded — it is entered with a wider CPT, which is exactly what a CPT is
for.

## 6. Direction — what has to be built, with this forecast alone

1. **Pin every locus to a tag.** Each diagnostic runs against the immutable
   `cycle-<date>_0000` tag; the CPT, the posterior and the grade all record it. `repo.ancestry()`
   is the audit trail. This is the single non-negotiable step — without it the network produces
   plots, not evidence.
2. **Ship L1→L2 for the grade-A path first:** IVT / AR probability (§3 of the diagnostics note)
   and `z500` regime clustering (§5). Neither needs a climatology.
3. **Discretize deliberately.** Every L2 node needs a state definition (terciles, quintiles, or
   physical thresholds such as `|IVT| > 250 kg m⁻¹ s⁻¹`). Write the thresholds down once, version
   them, and never re-tune them per event — a re-tuned threshold destroys the audit trail.
4. **Accumulate a model climatology.** Store `u850, u200, χ200, skt, z500, tcw, swh` per cycle
   (a few MB) so grade-C nodes become true anomalies within a season, and so CPT counts have
   more than one cycle behind them.
5. **Add the reduced full-window field set** (0–792 h, 12-hourly, a handful of fields) to lift
   grade-D nodes into the network.
6. **Ingest the observational side (§9).** CHIRPS/IMERG antecedent accumulations and the CHIRPS
   climatology, matched to each archived cycle's valid window. This is what makes the CPTs
   grounded rather than self-referential, and it is independent of steps 2–5 — start it in
   parallel.
7. **Close the loop back to the score.** Because the same store feeds the AI-WQ quintile
   submission and these diagnostics, the network can answer *"why did the model favour this
   quintile?"* from the ingredients — the justification narrative the CRMA audit entry needs.

## 7. Generating the CPTs from the Icechunk store

The store is a natural CPT factory: it is already a large sample of jointly-consistent
atmospheric states (50 members × 61 steps × 542 080 cells), lazily addressable. Five routes,
in increasing order of what they demand:

### Route A — Ensemble frequency counting (single cycle)

Discretize each L1/L2 diagnostic into states, then count co-occurrence across the member
dimension. `P(child = j | parents = k)` is the fraction of members in parent-state *k* whose
child falls in state *j*.

```python
import icechunk, xarray as xr, numpy as np
repo = icechunk.Repository.open_or_create(
    icechunk.local_filesystem_storage(".../20260730_0000/icechunk_v2"))
ds = xr.open_zarr(repo.readonly_session(tag="cycle-20260730_0000").store,
                  consolidated=False, zarr_format=3)          # lazy, 50 members

# parent  = regime state per member (L2), child = area-mean tp tercile over admin-1 (L3)
regime = classify_z500(ds["z_500"])            # -> (member, time) int in {0..K-1}
tp_ter = tercile(ds["tp"].sel(values=mask).mean("values"))   # -> (member, time) int in {0,1,2}

K, J = regime.max().item() + 1, 3
counts = np.zeros((K, J))
for k in range(K):
    for j in range(J):
        counts[k, j] = ((regime == k) & (tp_ter == j)).sum().item()
cpt = (counts + 1.0) / (counts + 1.0).sum(axis=1, keepdims=True)   # Laplace smoothing
```

**What this is, honestly:** a *within-cycle* conditional frequency. The 50 members carry 50
distinct IFS-perturbed initial conditions, so this is a real sample of initial-condition
uncertainty — but they are all conditioned on **one analysis and one flow regime**, and their
spread is inherited from the IFS perturbation rather than generated by AIFS. The effective
sample size is therefore below 50 (members sharing a synoptic situation are correlated), and
the table describes *how this forecast's uncertainty is structured*, not a climatological
relationship. Record both facts at the node: `n_members = 50`, `n_cycles = 1`,
`perturbation = IFS ENS (enfo/waef)`.

### Route B — Multi-cycle accumulation (the climatological CPT)

Repeat Route A across cycles and pool the counts. Each weekly cycle adds a genuinely new
initial condition, so the counts converge toward a model-climatological CPT. Two practical
levers:

- **Spatial pooling** — pool N320 cells within an admin-1 unit or a homogeneous climate zone
  before counting, so sparse parent-state combinations fill in.
- **Dirichlet smoothing** — treat each CPT row as Dirichlet(α), with α from Route D's elicited
  prior rather than a flat Laplace 1. The posterior row is `(α + counts) / Σ(α + counts)`,
  which degrades gracefully when a parent combination is rare.

Store the accumulated count tensor, not just the normalized CPT — counts are re-poolable, a
normalized table is not.

### Route C — Canonical (parametric) CPTs for wide parent sets

An L3 node with 5 parents at 3 states each needs 243 rows; the store will never fill that
honestly. Use a **canonical form** instead of a full table:

- **noisy-OR / noisy-MAX** — one "how strongly does this parent alone raise the risk"
  parameter per parent, from which the full table is generated. Interpretable, elicitable, and
  fits the ingredients logic ("any one of these can deliver the rain").
- **logistic / softmax CPT** — fit coefficients on continuous L1 diagnostics
  (`|IVT|`, jet index, χ200) directly, skipping discretization loss, then read off the table.

This is the route that scales; prefer it wherever a node has more than three parents.

### Route D — Elicited prior, updated by the ensemble

Forecaster judgement writes the initial CPT ("given a positive-IOD circulation, dry is roughly
2:1 over wet"); ensemble counts from Routes A/B update it as Dirichlet evidence, discounted by
the effective sample size. This is the CRMA position exactly — the network is not learned from
scratch, it is a **written-down belief that the store revises**, and both the prior and the
revision are in the audit trail.

### Route E — The leaf loci cannot come from the store

`P(impact | hazard state, exposure, vulnerability)` is not in any forecast. It comes from
EM-DAT records, the event storylines, and CLIMADA impact functions. Be explicit about the
seam: **upper network learned from the store, lower network elicited and storyline-informed.**
Mixing the two silently is the failure mode that makes a Bayesian network look objective when
half of it is judgement.

### Provenance rule for every route

A CPT is stored with: the tag(s) it was counted from, the discretization thresholds, the
smoothing prior, the effective sample size, and the reliability grade (A–D) of each parent.
That record is what makes a posterior *justified* rather than merely computed.

## 8. Ocean and wave variables as loci

AIFS-ENS v2 carries **11 wave fields** the v1 configuration did not —
`swh, mwp, cdww, mwd` (stored as `cos_mwd`/`sin_mwd`, recombinable as
`atan2(sin_mwd, cos_mwd)`), the period-banded partitions `h1012, h1214, h1417, h1721, h2125,
h2530`, and `wmb` — plus `skt`, which over ocean is the SST proxy. Wave fields are
sea-only; land cells must be masked (the inverse of the LSM mask applied to `sd`/`swvl`).

### How they enter the synoptic reasoning

| Field | Phenomenon it speaks to | Role as a locus |
| --- | --- | --- |
| `swh` + short-period partitions | **wind-sea** — locally forced; over the Somali/Arabian basin it tracks the **Somali low-level jet** | a semi-independent corroborating parent for the jet node, derived from a different prognostic than `10u/10v` |
| `mwp`, `h1721`, `h2125`, `h2530` | **long-period swell** — remotely generated, Southern Ocean / distant storms | arrival timing at the East African coast; a lead-time signal no wind field gives |
| `mwd` (from cos/sin) | **swell provenance** | discriminates S/SSW southern-swell from SE/NE monsoon wind-sea — the discriminator that makes the two loci above separable |
| `cdww` | **air–sea coupling / surface roughness** | modulates the wind-stress and evaporation feeding the moisture budget; a covariate on the IVT node |
| `swh` + `msl` minima, SW Indian Ocean | **tropical cyclone** signature | TC presence/track spread as a parent to both coastal hazard and moisture-steering loci |
| `skt` over ocean | **IOD** (west 50–70°E, 10°S–10°N minus east 90–110°E, 10°S–0°) and **Niño-3.4** | the boundary-forcing node — grade B (proxy) and grade C (no baseline) |
| `swh` at the coast | **coastal inundation, fisheries safety** | a genuine L3 impact locus in its own right, for the Somali/Kenyan/Tanzanian coast |

Two things make this more than an extra variable list. First, the wave state is a **memory
field**: swell integrates wind forcing over days and thousands of kilometres, so it carries
information about the large-scale circulation that an instantaneous wind field does not —
useful precisely in the day-18–33 window. Second, it gives the single-source network a form of
**internal corroboration**: two loci derived through different physical paths (wind-sea height
vs. 10 m winds) can agree or disagree, and disagreement is epistemically informative even
though both come from the same forecast.

### Can it be evaluated? Yes — with stated limits

- **Internal consistency (available now).** Check wind-sea `swh` against the 10 m wind field
  through a fetch/duration relation, per member. Systematic divergence is a model-coherence
  finding; per-member scatter is an uncertainty estimate. Needs no external data.
- **Cross-member coherence.** Does swell arrival timing cluster, or is it flat across 50
  members? A flat distribution means the locus carries no information this cycle — a legitimate
  and reportable epistemic result.
- **Against ERA5 waves** for the day-18–33 valid window, once cycles accumulate — the natural
  baseline, and the same source that would supply the grade-C anomaly baselines.
- **Against satellite altimetry** (Sentinel-3/6, CMEMS L3 significant wave height) for
  independent verification over the western Indian Ocean.
- **Limits to state plainly:** in-situ wave observation in the western Indian Ocean is very
  sparse, so verification is essentially reanalysis- and altimeter-based; the wave fields were
  also *inputs* to the model (stream `waef`/`wave`), so they are not an independent
  measurement; `cos_mwd`/`sin_mwd` are absent from the GRIB path and exist **only** in the
  Icechunk store; and the 432–792 h window means swell propagation cannot be tracked from
  generation.

## 9. Past observations — CHIRPS / IMERG and station data as loci

The forecast is the only *forecast*. It is not the only evidence. Satellite rainfall
(**CHIRPS**, ~5 km daily/pentadal, 1981→present; **IMERG**, 0.1° half-hourly, 2000→present) and
scattered **synoptic / airport station** reports (wind, temperature, humidity, pressure, present
weather) are **hard evidence** in the Part I sense — what we *measure* — and they should be in
the network. They do not compete with the forecast; they occupy four distinct roles it cannot
fill.

| Role | Observation used | Where it attaches | Why the forecast cannot do it |
| --- | --- | --- | --- |
| **R1 — Antecedent condition** | CHIRPS/IMERG accumulations over the preceding 10–90 days; station rainfall | a parent node beside L3 (wet/dry soil, catchment wetness, vegetation stress) | the store starts at hour 432; it holds no past. Antecedent wetness is what decides whether the same rainfall floods |
| **R2 — Climatological baseline** | CHIRPS 40-yr climatology, per pentad/dekad | fixes the **grade-C** loci — anomaly and percentile definitions for `tp`-like nodes | we hold the AI-WQ 20-yr climatology only for `tp/tas/mslp`; CHIRPS gives a far longer, regionally-tuned rainfall baseline |
| **R3 — Verification → CPT likelihood** | CHIRPS/IMERG for the *valid* day-18–33 window, matched to past cycles | supplies the counts for `P(observed outcome \| forecast locus state)` | this **is** the CPT the forecast cannot self-generate; a model cannot verify itself |
| **R4 — Present-state corroboration** | airport/synoptic wind, pressure, present weather | a second, independent parent for the L2 circulation nodes (Somali jet, ITCZ position) | breaks the single-source dependency — genuinely independent of the model chain |

**R3 is the important one.** It converts the network from *self-referential* (all CPTs counted
from model members) to *grounded*: the arrow from an L2/L3 forecast locus to the hazard node is
parameterized by how that locus has actually verified against CHIRPS in past cycles. This is
also the only route to an honest statement about the under-dispersion noted in §5 — a
rank-histogram or reliability diagram of past member counts against CHIRPS tells you how much
to widen every CPT row.

### Typing them honestly

Hard does not mean certain. Each observation source carries its own uncertainty, and the CPT
must reflect it:

- **CHIRPS** — station-blended infrared; good at pentadal/dekadal totals and drought monitoring,
  weaker on daily extremes and over the highlands. Strong for R1/R2, cautious for flood peaks.
- **IMERG** — microwave/IR, sub-daily; better temporal resolution for flash-flood onset,
  known bias over complex terrain and for light rain. Strong for R1 timing.
- **Stations** — near-truth at a point, but **sparse and spatially unrepresentative** across the
  Horn; report gaps are themselves informative. Enter as a point-value node with an explicit
  representativeness error, never as an area-mean.

That gives a three-tier hard-evidence grading that sits alongside the A–D derivation grading of
§5 — the two axes together are the "evidence quality" field the CRMA audit entry needs.

### Can it be re-evaluated? Yes — and this is where evaluation becomes possible at all

Everything in §8 was internal consistency, because a single forecast has no external referent.
Adding observations makes real evaluation available:

1. **Retrospective matching.** For each archived cycle tag, extract the L2/L3 loci over the
   day-18–33 valid window and pair them with CHIRPS/IMERG for the same window. Accumulating
   these pairs *is* Route B, now against observations rather than model counts.
2. **Reliability and rank histograms.** Are "35 of 50 members wet" cases wet 70 % of the time?
   This produces a **calibration correction** applied at the CPT, keeping the model output
   untouched and the correction auditable.
3. **Storyline hindcasts.** Run the diagnostics on cycles covering known EM-DAT events; the
   observed impact closes Route E's leaf CPT with real counts instead of pure elicitation.
4. **Station cross-checks on circulation**, not rainfall — airport wind roses against the
   Somali-jet index are a low-cost, independent test of an L2 locus.
5. **Honest limits:** CHIRPS/IMERG are themselves estimates (satellite + gauge blend), station
   density in the Horn is low and unevenly distributed, and archived cycles are few — early
   verification will have wide confidence intervals and must be reported with them.

**One rule.** Observations enter the network as evidence and as CPT parameters — never as a
silent post-hoc adjustment to the forecast fields. The moment a correction is applied without a
node and a record, the audit trail that justifies the whole framework is broken.

## 10. What this buys, in one line

The Icechunk store turns a single forecast from **one soft evidence node** into a **structured
chain of loci with counts attached**; past observations then turn those counts from
self-referential into **grounded**. Because every count is pinned to an immutable tag, the
resulting CPTs are reproducible, inspectable, and challengeable. That is the difference between
a model output and evidence.

---

*Sources: `main.tex` — "From Impact-Based Forecasting to Epistemic Risk Assessment", CRMA |
CRAF'd | ICPAC; `ICECHUNK_S2S_DIAGNOSTICS.md`; `README.md` (v2 field set);
`ecmwf_opendata_pkl_input_aifsens_v2.py` (per-member IFS-perturbed initial conditions).*
