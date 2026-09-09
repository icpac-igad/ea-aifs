# Handover — S2S process BN + the turn to explanation-first, for second opinion

**Date:** 2026-08-06 (rev 2) · **Author:** this session (Claude Opus 5, driven by @nishadhka)
**Status:** committed · Python side runs on real data · **Julia side unrun**
**Purpose:** hand this to a reviewer. Everything below is stated so it can be *attacked*.
Claims are numbered; if you disagree, reply by number.

*Rev 2 adds §4.6 — a scope change adopted after the BN was built. If you only have time for one
thing, review that.*

---

## 0. TL;DR for the reviewer

We adapted two ICPAC bn-ibf reference implementations (a Julia/RxInfer flood BN and its Python
icechunk data-prep) to the **AIFS-ENS v2 S2S Icechunk store** (50 members, native N320, hours
432–792), following the grouped process ontology in `qd-1.md.txt`. Then — after seeing what
broke — we **demoted risk and made circulation explanation the primary product**.

Four things most worth a second opinion, in order:

1. **Was the pivot right?** Explanation-first, risk downstream. §4.6 + `DIRECTION_EXPLANATION_FIRST.md`
2. ~~Does the explanation product survive its own ESS?~~ **TESTED — it did not, and production
   changed.** k = 4 clustering is unstable (subsample ARI 0.56); k = 2 is stable (0.82).
   Production moved to k = 2 across Python, Julia and the registry. Review whether the test
   itself is sound and whether the *remaining* diagnostics need the same treatment. §5.1
3. **Is the evidence wiring circular?** The chain CPTs are counted from the same 50 members that
   supply the evidence. We argue it's the correct null; it is the most attackable design call. §4.3
4. **Are the thresholds defensible?** Every judgement number is in one table. §6

## 1. What existed before this work

Committed earlier (`730a52c`, `e4bfe45`, `0ed0521`):

| File | What it is |
|---|---|
| `epistemic-reasoning-summary.md`, `main.tex` | the CRMA/epistemic framing — loci, CPTs, evidence typing |
| `ICECHUNK_S2S_DIAGNOSTICS.md` | what the store enables beyond skill scores |
| `CPT_BUILD_CRITIQUE_AND_PLAN.md` | critique of both + 5-phase plan (Part 0 scale×lead; A1–A6, B1–B8, C1–C6) |
| `NETWORK_DIRECTION.md` | Phase-0 contract: generative direction, latent `H`, evidence as children `P(E\|H)` |
| `discretization_registry.yaml` | Phase-0 contract: versioned state definitions |
| `cpt_build.py`, `cpt_artifact_20260730.nc`, `IMPLEMENTATION.md` | Phase-1 counter; the ESS ≈ 1.85 finding |

New work sits **on top of** those contracts. Please check it actually does — in particular that
the generative direction holds (we believe it does: `RO` hangs off `W` as a child, never a parent).

## 2. What this session added

| File | Status |
|---|---|
| `DIRECTION_EXPLANATION_FIRST.md` | **the scope decision** — read first |
| `S2S_BN_ONTOLOGY.md` | ontology ↔ verified store variables; carries a scope banner pointing at the direction note |
| `s2s_bn_evidence_prep.py` | **runs on real data**; primary path is now `--explain-out` |
| `k_regimes_test.py` + `k_regimes_test_20260730.json` | **runs**; the stability test that moved production from k=4 to k=2 |
| `mrp_stability_test.py` + `mrp_stability_20260730.json` | **runs**; the M/R/P state-resolution test — 4-state partitions unresolved |
| `low_level_jets.py` + `low_level_jets_2026072{3,30}.json` | **runs on both cycles**; Somali/Turkana jets by physical core speed |
| `upper_jet_and_highs.py` + `upper_jet_and_highs_2026072{3,30}.json` | **runs on both cycles**; TEJ + Mascarene/St Helena highs |
| `CROSS_CYCLE_FINDINGS.md` | what replicates on a second cycle and what does not — read alongside §5 |
| `s2s_water_balance_bn.jl` | **never executed** (no Julia on this box); re-typed as downstream consumer |
| `discretization_registry.yaml` | 0.1.0 → 0.2.0: `process_loci`, `blocked_loci`, circulation regions, `explanation_products` |
| `README.md` | folder arc extended with items 5–7 |

Commits: `617864a` (file move), `f964700` (the qd-1 brief), `a988c44` (BN implementation), plus
the commit carrying this revision. Inspiration: `icpac-igad/bn-ibf@jua-bnet` —
`flood_bn_ibf_v1.jl`, `flood_data_prep.py`.

## 3. Verified facts — please re-verify, do not take on trust

Probed directly against `/tank/projects/aifs-run/20260730_0000/icechunk_v2`, tag
`cycle-20260730_0000`, using `/tank/projects/micromamba/envs/aifs-gpu/bin/python`:

| Fact | Value |
|---|---|
| Shape | `member=50`, `time=132` allocated, **61 written** (idx 71–131 → 432–792 h, 6-hourly), `values=542080` |
| Vars | 122 data vars; `t,u,v,w,z` on 14 levels, `q` on 13 (no 10 hPa) |
| **Absent** | `cape`, `tcwv`, `lsm`, `z`/`slor`/`sdor` (surface orography constants) |
| **Present, unexpectedly useful** | `ro` (runoff), `sf`, `snowc`, `swvl1/2`, `cp`, `w` at all levels |
| Flux semantics | `tp, cp, ro, sf, ssrd, strd` are **interval per-6h** — established by flat field-means across lead time. The naive "fraction of cells decreasing" test is **inconclusive** for `ro`/`sf` (only 5.8 %/8.2 % of cells nonzero, so ties dominate) |
| Land mask | `finite(swvl1)` — 68.3 % NaN globally; IGAD box 11 342 cells, 8 780 land |

If any of these are wrong, most of §4 collapses. **Check this first.**

## 4. Design decisions — rationale *and* the case against

### 4.1 Target node = catchment water-balance pressure, not flood risk

*For:* the first written step is hour 432; event-scale flood reasoning at day 18–33 is not
defensible, and qd-1 agrees for the 360–800 h band. One ordered node serves both flood and
drought, matching ICPAC's dual mandate.
*Against:* harder to act on than a flood probability, and not what the bn-ibf downstream
expects — it forces a further, unmodelled step.

### 4.2 Chain DAG instead of the original's flat multi-parent CPT

*For:* bn-ibf's parents are near-independent summaries of one field (TP); ours are all functions
of the same model state, so as siblings they multiply-count one piece of evidence (qd-1's
"avoid this structure"; critique item A5).
*Against:* a chain asserts a sequential mechanism. Convection isn't sequential, and our
`P(P|M,R)` canonical form imposes an interaction shape nobody validated.

### 4.3 ⚠ Where evidence attaches — the circularity question

Each member gives one state at every node, so 50 joint draws. We use that twice: to **count**
`P(M|C)`, `P(R|M)`, `P(P|M,R)`, and to supply the **evidence** at `C`.

*Objection:* applying a CPT counted from sample *X* to evidence from sample *X* is circular; the
posterior just reproduces the ensemble's joint marginal and the BN adds nothing.
*Our answer, to be stress-tested:* that is the correct **null**. The forecast-side chain only
transports the ensemble's joint structure; the BN's value is where non-forecast information
enters (`A`, and the elicited `P(W|P,A)`, `P(RO|W)`). If you conclude the forecast side should
collapse to a single node, we want that argued — under the §4.6 pivot it may be the right call
anyway.
*Mitigation:* `--evidence-mode chain` (default) attaches ensemble evidence only at `C`, `A`,
`RO`; `--evidence-mode all` reproduces the naive port and **warns**.

### 4.4 Runoff as an evidence child, not a target

`ro` exists but the model has no routing and no real catchment, so it hangs off `W` as
`P(RO|W)` with a deliberately flat likelihood (self-test asserts the posterior shift < 0.30).
*Against:* one could argue it deserves more weight as the most direct catchment signal — or that
unrouted bucket runoff has no place in the network at all.

### 4.5 Absolute thresholds instead of within-ensemble quantiles — a bug found by running it

The first draft used within-ensemble quantiles ("no climatology, so make everything relative").
**Wrong:** a quantile state is a *rank*, so state fractions are fixed by the cut points
regardless of the forecast. Empirically every quantile-defined node returned identical soft
evidence for both lead windows (`A = [0.2]×5`) — zero information. Only the k-means circulation
node varied, because it classifies a *pattern*. Replaced with absolute physically-anchored
thresholds (§6); after the fix the evidence tracks the forecast (R: 0.72 → 0.86 supportive).
*Reviewer check:* sound replacement, or did we trade informationless for arbitrary?

### 4.6 ⚠ NEW — the pivot: explanation first, risk downstream

Adopted after the BN was built, on the evidence of what broke. Full argument and falsifiers in
`DIRECTION_EXPLANATION_FIRST.md`; the short version:

*For:* (a) every blocking dependency is on the risk side — no CHIRPS/IMERG, no EM-DAT, no
exposure/vulnerability, no elicitation panel — plus three structural defects (§5.2, §5.3, §5.4);
the explanation layer needs only the store and ERA5. (b) Part 0 of our own critique says
admissibility follows scale, and admin-1 pooling projects a planetary-scale signal onto polygons
where predictability is weakest. (c) *"n of 50 members"* framing is native to explanation and
fatal to calibrated risk — the same property that caused the §4.5 bug. (d) The store's unique
asset is the full 3-D state the submission discards.

*Against — argue any of these:* the customer may actually need an actionable trigger, not a
narrative; ICPAC's mandate is anticipatory action, and an explanation product may be a research
artefact; and the pivot's cost is real — χ200, MFC and vorticity now require a regrid or
spherical-harmonic step that was previously deferrable.

*Implemented, not just declared:* `--explain-out` writes the circulation record; the registry
carries the new circulation regions and the blocked-loci priority change; the Julia BN is
re-typed as downstream and warns at runtime.

## 5. Known issues we did not fix — please confirm or dispute

1. **✅ RESOLVED — `k = 4` was over-partitioning; production is now `k = 2`.**
   `k_regimes_test.py` clusters two independent 80 % member subsamples per *k* and compares
   them on the intersection with the Adjusted Rand Index (chance-corrected, so no null model
   needed). Mean over 6 cases: **k=2 → 0.817 stable · k=3 → 0.619 marginal · k=4 → 0.557
   UNSTABLE · k=5 → 0.544 · k=6 → 0.535.** Silhouette falls monotonically 0.43 → 0.22; PC1
   alone holds 72–88 % of variance. `loc.circ.v2` supersedes `loc.circ.v1`, counts never pool
   across them, and the Julia BN's `C` arity followed (4 → 2). Artifact:
   `k_regimes_test_20260730.json`.
   *Still open for the reviewer:* (a) is the ARI-on-subsamples protocol the right stability
   test here, or would you prefer a different one; (b) **one case dissents** — EQ_INDIAN week
   4–5 prefers k = 3 (k=2 ARI 0.53 there).

1b. **✅ ALSO RESOLVED — M / R / P were tested too, and they failed as well.**
   `mrp_stability_test.py` (artifact `mrp_stability_20260730.json`). The ARI protocol does not
   transfer to threshold-defined states — a member's state is a deterministic function of its
   own value, so subsample re-labelling is trivially perfect and measures nothing. Tested
   instead: degeneracy, cut fragility (density at the cut ÷ peak density), threshold
   sensitivity (TVD under ±10 %/±25 % cut shifts), sampling error (bootstrap at n=50 and
   n=ESS). Result over 6 cases: **M occupies 1.33 of 4 states** (4 cases fully degenerate),
   **R and P occupy 2.0 of 4**, with 4 of 6 cases FRAGILE each. Cuts inside the ensemble range
   sit almost on the density peak (up to **0.99**, 11 of 50 members within 5 % of the spread of
   the line). At n_eff the 90 % CI on a fraction is **0.33–1.00 wide**.
   *Action:* thresholds deliberately **NOT** re-tuned (that is failure C3 — fitting the
   partition to one event). Instead every record carries a `state_resolution` flag, the product
   leads with ranges + named-threshold exceedance counts, and the case for the model
   climatology (E4) is now the strongest item in the roadmap.
   *Open for the reviewer:* is "don't re-tune, flag it, and prioritise the climatology" the
   right response, or should the state vocabularies be cut to 2–3 states now?
2. **`tp` region-meaned over all cells; `ro`/`swvl` over land only.** Inconsistent; dilutes
   precipitation with ocean cells for box units. Moot for admin-1. *We consider this a defect.*
3. **The antecedent node is not independent of the forecast.** The fallback `A` uses the model's
   own `swvl1/2` from the *same run* that produces `P`, so `P(W|P,A)` treats two quantities from
   one integration as separate parents. The ontology note documents the *staleness*; this
   *dependence* is arguably worse and is documented only here.
4. **The circulation ordering index is season-dependent.** Clusters are ordered by
   `z(−msl_anom) + z(v_850)`; a southerly 850 hPa anomaly is moisture-bearing for East Africa in
   some seasons, not others (the Somali jet reverses). The ordering may invert MAM ↔ OND.
5. **Evidence is not ESS-discounted, only the counts are.** `C` state fractions enter at full
   strength as if 50 independent members.
6. **`kmeans2` can return empty clusters** (`missing="warn"`); the ordering step then sorts an
   `inf` mean. Not observed, not guarded.
7. **Quasi-equal-area assumption** — unweighted cell means on the reduced Gaussian grid. Worth a
   check by someone who knows the grid better than we do.
8. **Moisture node degenerate at box scale** (all 50 members `normal`, region-mean IVT 157–230
   inside the 150–250 bin). We call it a scale artefact and refuse to re-tune the threshold to
   split the sample; you may reasonably say the threshold set is simply wrong for regional means.

### 5.9 A defect found by challenge, not by us — the jet indices were weak

Raised by the user, and correct. The original `loc.somalijet.v1` took a **mean of `v_850`** over
a 15°×10° box and classified it **binary against the ensemble median**; the pipeline then
reported `n_southerly = (v > 0).sum()`, i.e. a sign test. Three defects, all of which we had
already diagnosed *elsewhere* and failed to apply here: a box mean dilutes a jet **core**; the
meridional component is not the literature criterion (that is speed |V|); and an ensemble-median
binary is a **rank**. Meanwhile the Turkana jet was registered `BLOCKED — needs orography`,
which was simply wrong: a geographic box plus a speed threshold needs no orography at all, and
only the *channelling attribution* does.

Both are fixed in `low_level_jets.py` / `loc.llj.v1` with literature thresholds (Somali 15/25
m/s at 850 hPa; Turkana 12/16.8/30 m/s at 925 hPa), a p95 core metric, direction checks and a
nocturnal 0300-local subset. *For the reviewer:* are the boxes right (Somali 40–55E/5S–10N;
Turkana 34.5–38.5E/1–6N), is p95-over-box the right core metric, and is 850 hPa the right level
for a jet quoted at "1.0–1.5 km ASL"?

## 6. Every judgement number, in one place (attack these)

**Python — `s2s_bn_evidence_prep.py` `THRESH`:**

| Quantity | Cuts | Stated anchor |
|---|---|---|
| region-mean \|IVT\| (kg m⁻¹ s⁻¹) | 150 / 250 / 350 | AR criterion (Guan & Waliser 250) at 0.6×/1.0×/1.4× |
| IVT persistence for `extreme` | ≥ 0.5 of steps above 250 | arbitrary-but-round |
| rainfall-env index | −1.0 / 0.5 / 2.0 | index = `ascent/0.01 + (lapse−26)/3 + (rh700−0.50)/0.15` |
| precip rate (mm day⁻¹) | 1.0 / 4.0 | round numbers, *not* climatological |
| episodic vs persistent | `peak24/total > 0.5` | scale-free shape rule |
| soil water (m³ m⁻³) | 0.12 / 0.20 / 0.28 / 0.35 | loam wilting point / field capacity |
| runoff rate (mm day⁻¹) | 0.05 / 0.25 / 0.75 | round numbers |

**Julia — the elicited lower network (Route E):** `M_SCORE`, `R_SCORE`, `P_SCORE`, `A_SCORE`,
`W_CENTRE`; kernel widths `sigma` (1.0 for W1, **1.35 for W2** — the deliberate widening with
lead); the two named non-linearities in `P(W|P,A)` (**−0.45** heavy-rain-on-dry damping,
**+0.45** persistent-rain-on-wet amplification); the `w_pers` episodic/persistent split; the
`P(RO|W)` matrix; `cost_loss_ratio = 0.2` (Weingärtner & Wilkinson 2019) in the two-sided CRMA rule.

**Structural:** `k = 2` regimes (tested, §5.1); lead windows W1 = 432–576 h, W2 = 582–792 h;
regions `IGAD_EA` [−12,18]×[22,52], `EQ_INDIAN`, `WIO`, `CONGO` (registry).

## 7. What is tested, and what is not

**Tested (real data, `cycle-20260730_0000`):** the whole Python path — store open, masks, all
six diagnostics, Somali-jet index, state assignment, soft evidence, per-member sidecar, counted
CPTs, netCDF/JSON artifacts, the explanation record over multiple regions, and the k-regimes
stability test (whose result changed production). Sanity checks
that passed: counts sum to 50 (one state per member — critique rule A1); C occupancy
`[8,11,15,16]` matches the reported fractions; ESS 1.85/1.87 reproduces the earlier
`cpt_build.py` finding; unobserved parent rows fall back to the prior while observed rows carry
real structure (`M→R` = `[0.05, 0.274, 0.626, 0.05]`).

**Not tested:** *anything in Julia* — no Julia on this machine. The BN mirrors the reference
file's verified RxInfer idioms and ships a `--test` self-check (CPT normalisation, wet/dry
monotonicity, runoff-corroboration bound, lead-window widening) — **none of it has been run.**
Its default path (`infer_chain`) is exact tensor contraction and needs no RxInfer, so start
there. Also untested: the `--adm1` GeoJSON path (hand-rolled point-in-polygon, written because
geopandas/regionmask aren't installed) — no admin-1 file was available.

**No verification against observations of any kind.** No skill, no null model, no calibration.

## 8. Suggested review order

1. §3 — re-verify the store facts. If the interval semantics or the absent-field list are wrong,
   stop and tell us.
2. §4.6 — the pivot. Right call, or are we abandoning the actionable product?
3. §5.1 and §5.1b — the two stability tests. Are the protocols sound, and is "flag it, don't
   re-tune it, prioritise the climatology" the right response to unresolved state partitions?
4. §4.3 — the circularity question.
5. §5.2–5.4 — the defects we named but did not fix.
6. §6 — thresholds and elicited scores, ideally by someone with East-African forecasting
   experience rather than a modeller.
7. Run `julia s2s_water_balance_bn.jl --test` if you have Julia, and tell us what breaks.

## 9. Open questions for a human decision

- Does the duty forecaster want *"31 of 50 members, ESS ≈ 2"*, or a probability they can act on?
  This decides whether §4.6 was right.
- Should the forecast side collapse to a single node (§4.3)?
- Which spatial unit is canonical — admin-1, hydrological catchments, or circulation regions?
  The code supports arbitrary polygons; nobody has decided.
- Who owns the elicited numbers in §6? Currently *ours*, which is the wrong provenance for an
  operational network.
- Do we accept the model soil-moisture proxy for `A` in the interim, or block the risk track
  until CHIRPS/IMERG (Phase 2) or the reduced full-window field set exists?
- Is a regrid/spherical-harmonic step (for χ200, MFC, vorticity) worth funding now that it is
  the explanation product's main blocker?

## 10. Reproduce

```bash
# PRIMARY — circulation explanation record over several regions
python s2s_bn_evidence_prep.py \
    --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 --tag cycle-20260730_0000 \
    --regions IGAD_EA,EQ_INDIAN,CONGO --explain-out explain_20260730.json

# DOWNSTREAM — BN evidence + the risk network
python s2s_bn_evidence_prep.py ... --out evidence.csv --member-out members.csv \
    --cpt-json process_cpt.json
julia --project s2s_water_balance_bn.jl --test
julia --project s2s_water_balance_bn.jl --input-csv evidence.csv --output-csv wbp.csv \
    --cpt-json process_cpt.json --evidence-mode chain
```

Runtime ≈ 4 min per region (the 13-level IVT integral dominates). Python env:
`/tank/projects/micromamba/envs/aifs-gpu` (icechunk, xarray, zarr≥3, scipy, yaml, pandas —
**no** geopandas/regionmask). Julia deps required but never installed here:
`RxInfer, CSV, DataFrames, JSON3`.
