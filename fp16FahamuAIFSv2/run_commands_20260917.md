# AIFS ENS v2.0 Run Commands — 20260917 (tier B; the silent kill explained)

**Date:** 20260917 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.
**Outcome: ✅ submitted 6/6 on 2026-09-20**, ~6 h inside the window. Verification weeks
**2026-10-05** and **2026-10-12**; scoreable from **2026-10-12**.

> **The headline of this cycle is not the forecast — it is that the silent kill was finally
> caught in the act.** Step 2 died at member 31 for the third time in five cycles, and this
> time the evidence survived: `apt-daily-upgrade.service`. See §"The silent kill, solved".

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data availability pre-check | seconds | RUNNABLE — 9/9 sfc, 14/14 levels, 11/11 wave |
| **—** | purge `20260212` (oldest, **upload complete**) | ~8 m | **106 GB reclaimed** |
| **1** | pkl creation, 50 members (proto, **GCS**) | **1 h 50 m** (109.9 min) → **131.1 s/member** | 42 GB, 50/50, 0 failed |
| **2a** | tier-B rollout, members 1–30 | **2 h 26 m** → 292.2 s/member | **killed at member 31** |
| **2b** | resume, `--skip-existing`, members 31–50 | **2 h 10 m** → 390.2 s/member | **50/50**, both stores tagged |
| **3a** | regrid 432–792 h → 1.5° NetCDF (from the sidecar) | **8.8 m** | 2.1 GB, 50/50, 0 failed |
| **3b** | quintiles → AI-WQ NetCDF | **~2 m** | 6.7 MB |
| **3c** | live submit via ECBox | **~1 h 15 m** | **6/6 submitted, 0 failed** |

Step 1 came back to 131.1 s/member, in line with 20260820's 126.5 and 20260909's 130.3. That
makes **20260910's 363.1 s a one-off**, not the start of a trend — worth recording because the
20260910 doc could not yet say which it was.

The resume again ran slower per member than the first pass (390 vs 292 s) with nothing else on
the GPU. Same direction as 20260910 (403 vs 311 s). Two cycles is a pattern; cause unknown.

---

## The silent kill, solved

Three cycles have now lost a rollout mid-member with **no traceback, no OOM, no kill message
and an empty kernel log**: 20260820 at member 40, 20260910 at member 39, 20260917 at member 31.

This time the box stayed idle afterwards, so the evidence was still there:

```
apt-daily-upgrade.timer   LAST: Fri 2026-09-18 06:06:12 UTC
rollout log stops:              Fri 2026-09-18 06:06    UTC
```

Same minute. And on 2026-09-12, when 20260910 died at 06:21, `dpkg.log` records:

```
06:22:02  python3.12  libpython3.12t64  libpython3.12-stdlib  libpython3.12-minimal
06:22:10  libc6  libc6-dev  libc-dev-bin
```

**Unattended-upgrades replacing the Python runtime and glibc underneath a running Python
process.** `setsid` detaches from the controlling terminal but leaves the process inside the
systemd **user slice**, which an upgrade-driven restart takes down without emitting a
per-process journal entry — which is exactly why five cycles' worth of searching for a
traceback found nothing.

It also explains the member numbers. A rollout started in the small hours is in the high
30s/40s by 06:00; this one started at 03:36 and was only at 31.

### The fix (needs sudo — NOT applied)

```bash
sudo systemctl disable --now apt-daily-upgrade.timer apt-daily.timer
```

On a dedicated GPU box that is defensible: unattended upgrades of glibc and the Python runtime
are actively hostile to multi-hour jobs. Alternatives, weaker: `loginctl enable-linger ezra`
so the user manager persists, or never running a rollout across **06:00–07:00 UTC**.

Until it is applied, **check the timer before launching**:

```bash
systemctl list-timers apt-daily-upgrade.timer --all | sed -n 2p   # next fire
```

### Recovery is safe, and cheap

`--skip-existing` requires a member present in **both** stores at each store's own final step,
so the partially-written member is redone rather than accepted:

```
To process: 20 member(s); skipped 30
```

Member 31 was verified finite in both stores afterwards. Write the resume to a **separate
log** — the original is the only record of the first pass.

---

## Commands (as run)

Unchanged from [`run_commands_20260903.md`](run_commands_20260903.md) except that **3a can now
pass `--icechunk-tag`**, because the sidecar tags itself since `5554586`:

```bash
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260917_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_n320_aiwq \
    --source-grid n320 --icechunk-tag cycle-20260917_0000 --output-dir $BASE/nc_1p5deg
# -> 📖 Icechunk store: …/icechunk_n320_aiwq (tag=cycle-20260917_0000)
```

Post-run validation, both stores at the tag:

| check | O96 corpus | N320 sidecar |
|---|---|---|
| `msl` shape | `(50, 132, 40320)` → O96 | `(50, 132, 542080)` → N320 |
| arrays | 124 | 14 |
| stored steps | **132/132** (0–792 h) | **61/132** (432–792 h) |
| members 1 / 26 / **31** / 50 | finite | finite |

Pre-flight before 3c — `--dry-run` exercises neither the ECBox token nor the AI-WQ checks:

| check | result |
|---|---|
| **provenance** | `icechunk_n320_aiwq @ **cycle-20260917_0000**` |
| window open · team+model registered | pass |
| 6 arrays | `(5,121,240)`, finite, ∈[0,1], Σ=1.000000 |

---

## Disk: the squeeze is now the binding constraint

| | |
|---|---|
| before the cycle | 258 GB free — **a tier-B cycle needs ~256 GB**, so it did not fit |
| freed | `20260212_0000/icechunk_o96`, **106 GB** |
| after the cycle | **103 GB free (95 %)** |

**Which "oldest" mattered.** The two oldest stores were both O96 archive cycles for the
source.coop mirror, and only one was mirrored:

```
20260212:  DONE 377,064 uploaded, 48.4 GB, 0 failed   <- complete, safe to delete
20260514:  stopped at 68,979 / 805,110                <- incomplete, still the only source
```

Deleting 20260514 would have stranded an unfinished 805,110-object upload. Check the upload
record before taking "oldest" literally.

**The next cycle will not fit.** 103 GB free against ~256 GB needed. The standing candidate is
`20260827_0000/icechunk_v2` — **628 GB**, the old full-N320 shape, submitted and scoreable from
2026-09-21 off the server-side file plus the local `aiwq/*.nc`, so the store is not needed for
scoring. Clearing it returns the box to ~730 GB.

`cleanup_aifs_run.py` still cannot reclaim a tier-B cycle — `HEAVY_PREFIXES` matches neither
`icechunk_o96` nor `icechunk_n320_aiwq`.

---

## Outputs

| Path | Size | What |
|------|------|------|
| `…/20260917_0000/input_states/` | 42 GB | 50 pkls + 50 symlinks |
| `…/20260917_0000/icechunk_o96/` | **164 GB** | 124 vars × 50 × **132/132 steps**, tag `cycle-20260917_0000` |
| `…/20260917_0000/icechunk_n320_aiwq/` | **53 GB** | 10 vars × 50 × 61/132 steps, same tag |
| `…/20260917_0000/nc_1p5deg/` | 2.1 GB | 50 × 1.5° NetCDF, from the sidecar |
| `…/20260917_0000/aiwq/…_20260917_v2.nc` | 6.7 MB | the AI-WQ product, provenance by tag |
| `/tank/projects/{proto_gcs_20260917_all50,run_0917_tierb,run_0917_tierb_resume,run_0917_3a,run_0917_3b,run_0917_3c}.log` | — | step logs — **two** Step 2 logs |
| [`fp16FahamuAIFSv2_20260917.txt`](fp16FahamuAIFSv2_20260917.txt) | — | 3b + 3c transcript |

---

## The TS checkpoint fired during this cycle

20260820's week 1 verified 2026-09-07…09-13, and the observation arrived mid-cycle. Run
against the refreshed IBTrACS:

| basin | **observed** | our forecast | terciles | outcome |
|---|---|---|---|---|
| **ATL** | **0** | P(below) = **0.92** | 6 / 11 | ✅ confidently right |
| **NWP** | **0** | P(above) = **0.84** | 3 / 7 | ❌ confidently wrong |

The zeros are genuine, not missing data: all 77 TS-strength records that week sat at
lon −169.5…−117.3 — entirely North-East Pacific, which AI-WQ does not score.

**This changes the TS plan.** The ~4× "ATL low bias" was not a bias; it was a correct forecast
of a quiet Atlantic, called at 0.92. A detector-native climatology is aimed at the wrong term
for ATL. **NWP's over-count is the real defect.** One week is one sample — 20260827's week 1
(09-14…09-20) gives a second point from 2026-09-21. See
[`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md).

---

## Tropical-storm days — run before this store is purged

The tracker was run on this cycle on **2026-09-21**, contributing 100 samples to the
detector-native climatology:

```bash
$PY ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20260917_0000 \
    --init 20260917 --out $BASE/ts_days_probs_20260917_tracked.nc
```

| week | ATL mean | NWP mean |
|---|---|---|
| 2026-10-05 | 0.9 | 8.7 |
| 2026-10-12 | 1.1 | 7.6 |

ATL at 0.9 and 1.1 is the quietest of the five cycles, consistent with the two
verified-zero Atlantic weeks that preceded it.

> **Retention rule.** The store is **53 GB** and will be purged; this product is
> **~20 KB** and is the entire scientific value of the cycle for this target. It sits at the
> cycle root, which `cleanup_aifs_run.py` does **not** protect — so running the tracker is a
> manual pre-purge step, and forgetting it is unrecoverable. See
> [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md).
