# Two cycles, not one — what replicates

**Cycles:** `cycle-20260730_0000` and `cycle-20260723_0000` (identical schema: 50 members, N320,
61 written steps 432–792 h). Every earlier finding in this folder rested on **one** cycle; this
note records which survive a second. Artifacts: `k_regimes_test_2026072{3,30}.json`,
`low_level_jets_2026072{3,30}.json`, `upper_jet_and_highs_2026072{3,30}.json`.

**Seven cycles are on disk** (`20260625` … `20260806`), so two is still a small sample — but it
is the difference between "a property of one Tuesday" and "a property we have seen twice".

---

## 1. Replicates cleanly

**Low-dimensional ensemble spread.** ESS 1.25–1.72 on 20260723 against 1.29–1.87 on 20260730;
PC1 holds 75–89 % of variance (72–88 % before). The 50-member spread really does collapse onto
~1–2 effective directions.

**k = 4 is unstable.** Subsample ARI 0.565 on 20260723, 0.557 on 20260730 — and the ordering
k=2 > k=3 > k=4 > k=5 ≈ k=6 holds on both. The decision to drop k=4 is confirmed.

**The jets carry more information than the fields.** ESS on 20260723: Somali 3.0–3.4, Turkana
**5.0–6.9**; on 20260730: 2.7–3.1 and 6.3–7.0. Two to four times the regional-field ESS in both
cycles. This is the single most reproducible result in the folder.

**The Somali jet's lead-time decay.** Week 3 → weeks 4–5 duration above 15 m/s: 98 % → 73 %
(20260723) and 91 % → 63 % (20260730). Same sign, same rough magnitude, both cycles.

**The Turkana nocturnal maximum.** Core at 0300 local: **14.01 m/s** (20260723) and **14.04 m/s**
(20260730) — against all-hours means near 11.8. Two independent cycles agreeing to 0.03 m/s on a
physical behaviour nobody put in the diagnostic.

## 2. Weaker on the second cycle — state honestly

**k = 2 is *marginal*, not stable.** ARI 0.688 on 20260723 versus 0.817 on 20260730 — below the
0.75 convention on this cycle. k=2 remains the best available choice and the k=4 rejection is
untouched, but the earlier claim that k=2 is "stable" was one cycle's luck. **The honest
statement is that the circulation partition is marginal at k=2 and unusable beyond it.**

## 3. The TEJ — the recipe generalises

Applying the low-level-jet recipe (localised feature + percentile core + absolute threshold) to
`u_200`:

| | 20260730 W1 | 20260730 W2 | 20260723 W1 | 20260723 W2 |
|---|---|---|---|---|
| African sector core (p95 of −u) | 17.6–30.0 | 12.9–24.9 | 13.9–24.2 | 14.7–23.8 m/s |
| members reaching 25 m/s | **36/50** | 31/50 | **20/50** | 16/50 |
| Indian sector core | 24.0–35.4 | 22.0–32.0 | 19.4–30.0 | 18.5–29.7 m/s |
| members reaching 25 m/s | 50/50 | 50/50 | 45/50 | 49/50 |
| ESS | 2.8–3.0 | 3.5–3.6 | 3.1–4.1 | 3.5–4.3 |

Three things worth noting. **The thresholds discriminate** — 16/50 to 50/50 across cases, not
saturated and not degenerate. **The Indian sector is stronger than the African sector in all four
cases**, which is the correct physics (the TEJ core sits upstream over India and weakens
westward) and the diagnostic was never told this. And **the two cycles genuinely differ** — the
African TEJ is markedly stronger in 20260730 (36/50) than 20260723 (20/50), which is the kind of
cycle-to-cycle signal an explanation product exists to report.

## 4. The subtropical highs — half the diagnostic works

**Strength saturates and is useless as specified.** The literature thresholds (1020/1024/1028 hPa
central pressure) are reached by 50/50 members with 90–100 % duration in *every* case on *both*
cycles. The metric (box max over a large winter-hemisphere box) and the threshold (a
climatological central pressure) are mismatched — the box max picks up transient migratory highs,
not the semi-permanent centre.

**Position works and is informative.** Mascarene ridge centre sits at 32.4–33.4 °S, 67.7–69.2 °E
across both cycles, and the member spread is strikingly **anisotropic**:

| | latitude sd | longitude sd |
|---|---|---|
| Mascarene | 1.0–1.5° | **5.9–8.9°** |
| St Helena | 1.7–2.7° | **4.1–5.4°** |

The ensemble agrees closely on *how far north* the ridge sits and disagrees strongly on *how far
east* — consistently, on both cycles. Since ridge longitude sets the direction of the
cross-equatorial flow feeding the Somali jet, that is a physically meaningful disagreement and a
better discriminator than any strength threshold.

**Correction taken, not a re-tune.** The thresholds are left untouched: moving them to fit would
be the C3 threshold-re-tuning failure. The record carries a `threshold_caveat` naming the
mismatch, `ridge_strength_hPa` (box mean and p95) is reported alongside the saturating max, and
the proper fix is the model climatology (E4) so cuts sit at climatological percentiles.

## 5. What this changes

- **E4 (model climatology) is now the clear priority.** It is the fix for the M/R/P threshold
  mismatch *and* the highs' saturation — two independent lines of evidence pointing at the same
  missing piece.
- **The jet/TEJ family is the strongest explanation locus available** and should carry the
  product: high ESS, discriminating thresholds, physically consistent behaviour, replicated.
- **The circulation-regime node is the weakest** and should be reported with the marginal-stability
  caveat, or replaced by projection onto fixed reference patterns (which would also make labels
  comparable across cycles — currently they are not).
- **Run the other five cycles.** Two agreed on the important things and disagreed on one; five
  more are sitting on disk and would settle whether k=2 is usable at all.

---

*A methodological note worth keeping: a bug was found while adding these diagnostics — `360 %
360 = 0` silently emptied the St Helena box, producing zero cells rather than an error. It was
caught only because `argmax` then raised on an empty array. The shared `box_cells` primitive is
now wrap-aware. Any box touching the prime meridian was affected.*
