# Conditional probability tables from the Icechunk store and observations — critique and build plan

*A critical evaluation of `epistemic-reasoning-summary.md` and `../ICECHUNK_S2S_DIAGNOSTICS.md`,
followed by the corrected direction for building the CPTs.*

---

## Part 0 — The framing both documents under-specify: scale × lead time

Both notes reason carefully about *evidence quality* (grades A–D, hard/soft/virtual) but
neither states the physical fact that governs everything at days 18–33: **predictability is
scale-dependent, and the forecast horizon must be an explicit conditioning variable of every
CPT, not background context.**

At this lead:

| Scale | What survives to days 18–33 | How it may enter the network |
| --- | --- | --- |
| **Planetary / boundary-forced** (IOD, ENSO, MJO envelope) | genuine signal, largely carried in from the initial state | as slowly-varying parent nodes — the *source* of week-3+ skill |
| **Synoptic** (~1000 km: `z500` regimes, IVT plumes, Somali jet, ITCZ position) | probabilistic signal — regime *occupancy frequencies* are informative, individual member trajectories are not | as L2 loci, counted across members, **per lead window** |
| **Mesoscale** (~100 km: convective organisation, jet streaks, coastal circulations) | essentially none deterministically; only its *statistics conditioned on the synoptic state* | never as a direct evidence node; only inside a CPT row `P(local outcome \| synoptic state)` learned from observations |
| **Grid-wise** (N320, ~30 km cells) | none. A member's grid-point rain at day 25 is model-textured noise, not information | only after spatial pooling (admin-1 / climate zone) or as exceedance statistics; never cell-by-cell |

This gives the two rules the documents are missing, and which the Bayesian-network logic is
exactly the right machinery to enforce:

1. **Admissibility rule.** A locus is admitted at a given lead only if its scale retains
   predictability there. Grid-wise fields are *inadmissible* as direct evidence at days 18–33;
   they become admissible only as pooled statistics or through an observation-learned
   conditional (`P(admin-1 wet | regime B)` — a CPT row, which is where the mesoscale/local
   understanding lives).
2. **Lead-stratification rule.** Every CPT is indexed by lead window (e.g. the two AI-WQ-aligned
   weekly windows inside days 18–33, plus the remainder). Reliability decays with lead;
   one pooled table across 61 steps silently averages a week-3 relationship with a week-5 one.

This is also the honest **risk framing**: the network does not pretend to forecast a flood at a
grid cell in week 4. It states *"the large-scale pattern that historically delivers heavy rain
to this admin-1 unit is carried by n of 50 members in lead window W, and antecedent wetness from
CHIRPS is at the p-th percentile"* — a justified, auditable belief at the scale where belief is
justifiable, which is what the CRMA grade (L4) should consume.

---

## Part I — Critical evaluation

### A. Wrong or unsupported assumptions

**A1. Pseudo-replication in the Route A counting code (summary §7, code sketch).**
The sketch counts co-occurrence over the flattened `(member, time)` array: 50 × 61 = 3 050
"samples", then Laplace-smooths as if n = 3 050. Consecutive 12-h steps within one member are
strongly autocorrelated (a regime persists for days), and members sharing a synoptic situation
are cross-correlated. The effective sample size is tens, not thousands. As written, the CPT
will look far sharper than it is — the exact overconfidence the framework exists to prevent.
*Fix:* reduce each member to **one state per lead window** (modal or time-mean state) before
counting, so n ≤ 50 per window; then discount further for inter-member correlation.

**A2. The network arrows point the wrong way for evidence fusion.**
The summary defines L3 as `P(hazard | L2 forecast state)` (§5) and R3 as
`P(observed outcome | forecast locus state)` (§9), and then R4 adds station observations as "a
second parent" of the same circulation nodes. That structure cannot coherently combine
independent evidence: with forecast state and station state both parents of the outcome, you
need the *joint* table `P(outcome | forecast × station)`, which no realistic sample will fill —
and pairwise-calibrated tables cannot just be multiplied. The standard, and correct, structure
is **generative**: a latent hazard/circulation state `H` with evidence nodes hanging *off* it —
`P(forecast locus state | H)`, `P(station report | H)`, `P(CHIRPS antecedent | H)` — so the
posterior is `∝ P(H) · Π P(Eᵢ | H)` and each likelihood table is learnable independently from
matched pairs. R3's verification pairs are exactly the data for `P(forecast state | H)`; the
documents just wire them in backwards.

**A3. `tercile(ds["tp"]...)` contradicts the documents' own rules.**
Grade C (summary §5) says `tp`-like anomalies "must not be stated as an absolute index" without
a baseline, and R2 says CHIRPS climatology is what fixes them — yet the Route A sketch computes
a tercile with no stated baseline. Within-ensemble terciles and CHIRPS-climatology terciles are
*different state definitions*; mixing them across cycles quietly changes the meaning of every
CPT row. Also unverified: whether stored `tp` is accumulated-since-initialization (typical for
AIFS output). If so it must be de-accumulated (step-differenced) before any windowed statistic
— raw values at hours 432–792 are 18–33-day totals, not interval rain.

**A4. `skt`-based IOD/Niño nodes are mis-typed.**
AIFS-ENS has no ocean model; over ocean `skt` is essentially the initial SST persisted/weakly
evolved. Consequences: (i) the box-mean `skt` is unusable as an IOD/Niño index — **corrected
2026-08-07 by finally measuring it: the claim originally written here, that the member spread is
"near-degenerate", is WRONG.** On `cycle-20260730_0000` the member sd of box-mean `skt` is
0.18–0.26 K, and the member sd of the west-minus-east dipole is **0.32–0.39 K — as large as a
weak IOD event (~0.4 K)** — ranging −1.30…+0.27 K across members. That spread is *atmospheric
skin-temperature noise* (skin temperature responds to wind and radiation), not ocean-state
information. The conclusion stands but for the opposite reason: the index is **swamped, not
frozen**. Worse, the absolute west-minus-east difference is dominated by the **climatological**
warm-pool gradient, so without an anomaly baseline it cannot indicate phase at all; (ii) epistemically these are closer to **hard evidence observed at initialization**
(the analysis SST the model was handed) than to soft forecast evidence. They should enter as a
single boundary-condition node with observational uncertainty, not as a member-counted locus.

**A5. Wave "internal corroboration" is overstated (summary §8).**
In a physics model, wind-sea and 10 m winds come from separate prognostic systems, so agreement
is corroboration. In an ML model trained on a jointly-consistent reanalysis/analysis record, the
learned joint distribution *is* the correlation — agreement is expected by construction and
carries little confirmatory weight. Divergence is still a useful *model-coherence* diagnostic,
but the claim of "semi-independent" parent evidence double-counts one source. Same class of
error at network scale: IVT, `tcw`, moisture-flux convergence and `tp` are all functions of the
same `q,u,v` state; wiring them as independent parents of a hazard node multiply-counts a single
piece of evidence. The BN structure needs an explicit **derivation-independence audit** before
any arrow is drawn.

**A6. Route B convergence is oversold.**
"Counts converge toward a model-climatological CPT" — at one cycle per week, a season yields
~13 cycles, and East-African rainfall regimes demand seasonal stratification (MAM ≠ OND), which
divides the sample again. Furthermore all cycles share the model's systematic biases, so Route B
converges to the *model's* climate joint, not the climate. Both documents note limitations
piecemeal, but the build orders (summary §6, diagnostics §7) sequence work as if Route B and R3
bear fruit in weeks. They bear fruit in **years** unless accelerated (see Plan, Phase 3).

### B. Major missing points

**B1. Lead-time stratification** — absent from every CPT definition in both documents (see
Part 0). "Capture the forecast horizon" currently happens nowhere in the formalism.

**B2. A scale-admissibility rule** — spatial pooling appears only as a sample-size lever
(summary Route B); it is actually a *predictability necessity* at this lead (Part 0).

**B3. An effective-sample-size and discount policy usable from day one.** Under-dispersion is
acknowledged (§5.2) and rank histograms are promised "once cycles accumulate" — but nothing says
what to do *now*. Needed: a written α-policy — e.g. Dirichlet prior mass equal to a substantial
fraction of the member count, inter-member correlation estimated from the fields themselves,
ESS recorded in the CPT artifact — so early posteriors are wide by construction.

**B4. Observation-side engineering.** §9 names CHIRPS/IMERG but not: the upscaling protocol
(CHIRPS ~5 km → N320 ~30 km → admin-1 — conservative area-weighting, and *pool then classify*,
never classify at 5 km and vote); temporal alignment (pentads/dekads vs 12-h steps and the two
lead windows); and **latency** (CHIRPS final lags ~3 weeks; the preliminary product and IMERG
Early/Late differ from final — the R1 antecedent node in real-time operation uses a *different,
noisier product* than the one the CPT was trained on, and the node must say so).

**B5. A null model.** Nothing in either document defines how we would know the network adds
value. Required: the climatology-only network (priors, no forecast evidence) as baseline, and
Brier/ranked-probability skill of the L3/L4 posteriors against it, evaluated at the *grade*
level, not the grid.

**B6. The L4 seam has no process.** Route E states the store/elicitation seam honestly but
gives no elicitation protocol, no loss framing for Monitor/Evaluate/Assess/Actionable, no
record format for elicited priors. The audit trail claimed as the framework's core benefit
currently has no defined artifact to store (B7).

**B7. No CPT artifact schema.** The provenance rule (§7) lists what must be recorded — tags,
thresholds, prior, ESS, grades — but no format, storage location, or versioning mechanism.
Without a concrete schema this rule will be violated by the first script written.

**B8. Hindcast acceleration unexplored.** The only stated route to grounded CPTs is waiting
for weekly cycles to accumulate. Whether any AIFS-ENS re-forecast/hindcast set (or an IFS-ENS
extended-range hindcast as a structural stand-in) can pre-fill `P(forecast state | H)` deserves
one investigation before committing to years of accumulation.

### C. Failure points (where it breaks silently)

1. **Time-pooled counting** (A1) — runs fine, returns confident nonsense.
2. **State-definition drift** (A3) — within-ensemble terciles this cycle, CHIRPS terciles next;
   counts pooled across the change corrupt every row with no error raised.
3. **Threshold re-tuning** — warned against in §6.3, but with no versioned registry the warning
   is unenforceable.
4. **Evidence double-counting** (A2/A5) — the posterior sharpens as correlated diagnostics are
   added, looking like increasing confidence while it is only repetition.
5. **Real-time/training product mismatch** (B4) — CPT trained on CHIRPS-final, operated on
   CHIRPS-prelim/IMERG-Early.
6. **De-accumulation of `tp`** unverified (A3) — one wrong assumption poisons every
   rainfall-derived locus.

---

## Part II — Build plan: CPTs from the Icechunk store + observations

Restructured around A2 (generative direction), Part 0 (scale × lead), and honest timelines.
Phases 1–2 are parallel; nothing waits on Route B/R3 accumulation.

### Phase 0 — Contracts before code (this week)

- **Network direction fixed in writing:** latent node `H` = hazard-relevant circulation/rainfall
  state per admin-1 unit per lead window. All evidence — forecast loci, CHIRPS antecedent,
  stations — attaches as children: `P(E | H)`. Posterior by product of likelihoods. One page,
  versioned, cited by every script.
- **Discretization registry:** one YAML file, versioned in git — every locus's state definition
  (thresholds, baseline used, pooling region), with an ID that CPT artifacts must reference.
  Changing a definition mints a new ID; counts never pool across IDs (kills failure C2/C3).
- **CPT artifact schema:** netCDF (or JSON for small tables) holding the **raw count tensor**
  (never only the normalized table), plus: source tag list, discretization-registry IDs, lead
  window, Dirichlet α and its provenance (Route D elicitation record), ESS, parent grades
  (A–D), and the observation product+version for any obs-derived axis. Stored beside the run
  archive, committed like the stores are tagged.
- **Lead windows declared:** the AI-WQ-aligned weekly windows within days 18–33, plus the
  residual tail; every CPT carries exactly one.
- **Verify `tp` semantics** in the store (accumulated vs interval) on the 20260730 tag before
  any rainfall locus is coded.

### Phase 1 — Grade-A, single-cycle likelihoods, counted correctly (weeks 1–2)

- Loci: **IVT/AR probability** and **`z500` regime occupancy** (the two no-climatology,
  grade-A paths), plus the Somali-jet index labelled *relative*.
- Counting fix (A1): per member, reduce to one state per lead window (modal state); count over
  the member dimension only → n ≤ 50; apply the Phase-0 α-policy; record ESS.
- Derivation-independence audit (A5): document which loci share source fields; the network
  admits at most one representative per correlated cluster as an evidence node (e.g. IVT *or*
  moisture-flux convergence for the moisture-supply role, not both).
- Output: the within-cycle tables typed honestly as *structure of this forecast's uncertainty*
  (`n_cycles = 1` in the artifact), entering the network as soft evidence about `H`, never as
  the `P(E | H)` calibration (that is Phase 3's job).

### Phase 2 — Observation pipeline (parallel, starts now; independent of the store)

- **Ingest:** CHIRPS (daily + pentad, final and preliminary) and IMERG (Late) for the IGAD
  region; CHIRPS 1981– climatology per pentad/dekad.
- **Upscaling protocol (B4):** conservative area-weighted aggregation 5 km → admin-1 (and to
  N320 where cell-level pooling is needed); classify *after* pooling; protocol text lives in
  the discretization registry.
- **R1 antecedent nodes ship immediately:** 10/30/90-day accumulation percentiles vs CHIRPS
  climatology per admin-1 — these need no forecast matching at all and are the fastest
  grounded evidence into the network. Record which product (final/prelim/IMERG) fed each
  real-time evaluation (kills C5).
- **R2:** CHIRPS-climatology terciles/quintiles become the *only* sanctioned `tp` state
  definition (retires A3's within-ensemble tercile).
- Boundary-condition node (A4): IOD / Niño-3.4 from *observed* SST analyses (e.g. OISST) at
  initialization time — hard evidence, replacing the member-counted `skt` locus.

### Phase 3 — Matched archive → the real `P(forecast state | H)` (continuous, slow; be honest)

- For **every archived cycle tag**, extract Phase-1 locus states per lead window and pair with
  the CHIRPS outcome for the same valid window; append to a pair archive (the artifact of
  record — tables are re-derivable from it).
- Accumulation arithmetic, stated up front: one cycle/week × seasonal stratification ⇒ single
  digit-to-low-tens of pairs per season per region per year. Early tables are dominated by the
  Route-D elicited prior *by design*; the artifact's ESS field makes that visible.
- **One-off investigation (B8):** whether an AIFS-ENS re-forecast set, or the IFS extended-range
  hindcasts as a structural prior, can pre-fill these likelihoods; a week of work that could
  save two years of waiting.
- Rank histograms / reliability of member counts vs CHIRPS as pairs accumulate → the
  dispersion correction entering as a *recorded* widening of CPT rows (never a silent
  adjustment of fields).

### Phase 4 — Model climatology and stratified Route B (background accumulation)

- Persist the small per-cycle field set (`u850, u200, χ200, skt, z500, tcw, swh`) as already
  planned; grade-C loci become in-season anomalies when the seasonal sample exists.
- All Route-B pooling is **stratified by season and lead window**; the count tensors carry the
  stratum. No cross-season pooling, ever, for rainfall-linked loci.

### Phase 5 — Evaluation harness and the decision seam

- **Null model (B5):** climatology-only network; report Brier/RPS skill of L3/L4 posteriors
  against it each season. If the forecast evidence adds nothing at a lead window, the
  admissibility rule (Part 0) says so publicly — that is a feature.
- **L4 elicitation protocol (B6):** structured elicitation of
  `P(grade | H, exposure, vulnerability)` rows with named experts, date, and rationale in the
  artifact; storyline hindcasts on EM-DAT events as the first stress test.
- Every network run writes the CRMA audit entry: posterior, evidence list with grades and ESS,
  registry IDs, store tags.

### Way forward — first three concrete actions

1. Write the Phase-0 contracts (network-direction page, discretization registry, CPT artifact
   schema) — small documents, everything downstream depends on them.
2. Re-implement the Route-A counter on the `cycle-20260730_0000` tag with the A1 fix
   (per-member modal state per lead window, n ≤ 50, α-policy applied) for `z500` regimes →
   admin-1 pooled `tp` states — after verifying `tp` accumulation semantics.
3. Start the CHIRPS ingest and the antecedent-percentile (R1) node — the first grounded, hard
   evidence in the network, available regardless of how many cycles exist.

---

*Companion to `epistemic-reasoning-summary.md` and `../ICECHUNK_S2S_DIAGNOSTICS.md`; grounded in
the verified 20260730 store (120 vars, 50 members, N320, hours 432–792).*
