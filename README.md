# ea-aifs — AI Weather Quest forecasts from AIFS-ENS 2.0

Operational pipeline for **`fp16FahamuAIFSv2`**: a 50-member AIFS-ENS-2.0 ensemble run every
Thursday to 792 h (33 days), post-processed to weekly quintile probabilities and submitted to
the [AI Weather Quest](https://ai-weather-quest.ecmwf.int/).

**`fp16FahamuAIFSv2` is the only model still run.** `FahamuAIFSv1`, `fp16FahamuAIFSv1` and
`era5tFp16FahamuAIFSv1` are **deprecated** — kept for reference, not maintained. See
[Deprecated models](#deprecated-models).

| | |
|---|---|
| model | `ecmwf/aifs-ens-2.0`, FP16 |
| members | 50 |
| lead time | 792 h (33 days), 6-hourly |
| submitted | `tas`, `mslp`, `pr` × weeks 3 & 4 (init+18, init+25 days) |
| box | local RTX 5000 Ada, 30 GB — **not** Coiled |
| cadence | Thursday init; the AI-WQ window closes **init + 3 days, 23:59 UTC** |

---

## Weekly cycle records

**Every cycle produces two files, both in [`fp16FahamuAIFSv2/`](fp16FahamuAIFSv2/).** They are
the primary record — the run logs they are built from live outside the repo in
`/tank/projects/` and are not backed up.

| file | what it is |
|---|---|
| `run_commands_<DATE>.md` | the cycle's commands **as run**, timings, validation, and what went wrong |
| `fp16FahamuAIFSv2_<DATE>.txt` | verbatim console transcript of steps 3b + 3c, including the submission receipt |

| cycle | run doc | transcript | notes |
|---|---|---|---|
| 20260709 | [md](fp16FahamuAIFSv2/run_commands_20260709.md) | [txt](fp16FahamuAIFSv2/fp16FahamuAIFSv2_20260709.txt) | leaderboard `+0.071 / +0.055 / +0.106` (tas/mslp/pr, week 3) |
| 20260716 | [md](fp16FahamuAIFSv2/run_commands_20260716.md) | [txt](fp16FahamuAIFSv2/fp16FahamuAIFSv2_20260716.txt) | |
| 20260723 | [md](fp16FahamuAIFSv2/run_commands_20260723.md) | [txt](fp16FahamuAIFSv2/fp16FahamuAIFSv2_20260723.txt) | |
| 20260730 | [md](fp16FahamuAIFSv2/run_commands_20260730.md) | [txt](fp16FahamuAIFSv2/fp16FahamuAIFSv2_20260730.txt) | |
| 20260806 | [md](fp16FahamuAIFSv2/run_commands_20260806.md) | [txt](fp16FahamuAIFSv2/fp16FahamuAIFSv2_20260806.txt) | |
| 20260813 | [md](fp16FahamuAIFSv2/run_commands_20260813.md) | [txt](fp16FahamuAIFSv2/fp16FahamuAIFSv2_20260813.txt) | FTP → **ECBox** migration; 3c ~12 min/file |
| 20260820 | [md](fp16FahamuAIFSv2/run_commands_20260820.md) | [txt](fp16FahamuAIFSv2/fp16FahamuAIFSv2_20260820.txt) | O96 archive + N320 rerun — two stores, one cycle |
| 20260827 | [md](fp16FahamuAIFSv2/run_commands_20260827.md) | [txt](fp16FahamuAIFSv2/fp16FahamuAIFSv2_20260827.txt) | the `HF_HOME` trap |
| **20260903** | [md](fp16FahamuAIFSv2/run_commands_20260903.md) | [txt](fp16FahamuAIFSv2/fp16FahamuAIFSv2_20260903.txt) | **first tier-B cycle — 209 GB instead of 583 GB** |

How the transcript is assembled, and the two traps in doing so (ANSI escapes, and scanning for
credentials — 3c authenticates twice per file):
**[`RUN_LOGS_AND_TRANSCRIPTS.md`](fp16FahamuAIFSv2/RUN_LOGS_AND_TRANSCRIPTS.md)**.

---

## The cycle, end to end

Roughly **9 hours** of wall time, of which ~4.5 h is GPU. Start with the most recent
`run_commands_*.md` — it is the authoritative version of these commands.

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/<DATE>_0000
```

| step | what | time | output |
|---|---|---|---|
| **0** | is the cycle published? | seconds | `check_open_data_inputs.py --date <DATE> --source gcs` |
| **1** | input pkls from the GCS mirror | ~2.5 h | 42 GB, 50 members |
| — | **symlinks** — the step that breaks runs | seconds | `proto_input_state_member_NNN.pkl` → `input_state_member_NNN.pkl` |
| **2** | GPU inference → Icechunk | ~4.5 h | see [storage](#storage-tier-b) |
| **3a** | regrid 432–792 h → 1.5° NetCDF | ~9 min | 2 GB, 50 files |
| **3b** | quintile probabilities | ~2 min | 6.7 MB — the submission input |
| **3c** | submit via ECBox | ~1 h 15 m | 6 files (3 vars × 2 weeks) |

Three things that have each cost a run:

- **Create the symlinks.** The builder writes `proto_input_state_member_NNN.pkl`; the runner
  opens `input_state_member_NNN.pkl`. Count *and* resolve them — `ln -sf` links happily to a
  missing target. Skipping this makes step 2 exit in 2 seconds with `50 failed`.
- **`export HF_HOME=/tank/projects/hf_cache`.** Three plausible cache paths exist on the box
  and two contain an `aifs-ens-2.0` directory; only this one is complete. The wrong one does
  not fail — it stalls silently with the GPU at 0 %.
- **Run steps 2 and 3c detached** (`setsid nohup … &`) with `-u`. Both exceed any foreground
  timeout, and a SIGTERM mid-member left one cycle at 39/50.

### Storage: tier B

Since 20260903 a single rollout writes **two stores**, replacing the 583 GB full-N320 shape:

| store | grid | leads | variables | size |
|---|---|---|---|---|
| `icechunk_o96` | O96, 40 320 cells (~112 km) | **all 132 steps, 0–792 h** | 124 | 158 GB |
| `icechunk_n320_aiwq` | N320, 542 080 cells (~28 km) | 432–792 h | **10** | 51 GB |

**209 GB per cycle**, and the O96 corpus keeps days 8 and 15, which MJO needs and the old
`--write-hours 432-792` window discarded. 3a reads the **sidecar**; the corpus carries
everything else. Rationale and measurements:
[`O96-icechunk-store/README.md`](fp16FahamuAIFSv2/O96-icechunk-store/README.md) §7.

> The sidecar's 10 variables were chosen for the AI-WQ submission and the TS tracker. It holds
> **no `q`, no `w`, no `z`, and winds only at 850** — anything else at 28 km needs
> `--native-vars` widened, which is an **inference-time** choice that cannot be recovered
> without re-running the rollout.

### Before submitting

`--dry-run` short-circuits before the AI-WQ checks, so it exercises **neither the ECBox token
nor the validation**. What works offline — window open, team/model registered, all six arrays
`(5,121,240)` finite and summing to 1, and (since 20260903) `source_icechunk_store` confirming
which store built the file — is in
[`run_commands_20260903.md`](fp16FahamuAIFSv2/run_commands_20260903.md#step-3c--submit).

---

## Documentation index

**Operations**
- [`fp16FahamuAIFSv2/README.md`](fp16FahamuAIFSv2/README.md) — scripts, environments, the v2 scoping
- [`LOCAL_GPU_RUN.md`](fp16FahamuAIFSv2/LOCAL_GPU_RUN.md) — step 2 on the local box, environment build, and the HuggingFace-cache stall
- [`RUN_LOGS_AND_TRANSCRIPTS.md`](fp16FahamuAIFSv2/RUN_LOGS_AND_TRANSCRIPTS.md) — where logs live, how transcripts are built
- `cleanup_aifs_run.py` — reclaims a finished cycle ⚠️ **does not yet match tier-B store names**

**Storage & performance**
- [`O96-icechunk-store/`](fp16FahamuAIFSv2/O96-icechunk-store/) — the O96 route, tier B, manifest compaction
- [`ICECHUNK_PATH_A.md`](fp16FahamuAIFSv2/ICECHUNK_PATH_A.md) · [`ICECHUNK_COMMIT_CADENCE.md`](fp16FahamuAIFSv2/ICECHUNK_COMMIT_CADENCE.md) · [`LOAD_TEST_RESULTS.md`](fp16FahamuAIFSv2/LOAD_TEST_RESULTS.md)

**Verification**
- [`O96-icechunk-store/forecast-evaluation/`](fp16FahamuAIFSv2/O96-icechunk-store/forecast-evaluation/) — score a cycle locally against AI-WQ observations, using the competition's own code

**Research**
- [`ts-mjo/`](fp16FahamuAIFSv2/ts-mjo/) — tropical-storm-days tracker and MJO. AIFS-ENS 2.0 emits no OLR, so Wheeler–Hendon RMM is not computable; `vpm-mjo.md` proposes VPM as the AIFS-only route
- [`epistemic-reasoning-risk/`](fp16FahamuAIFSv2/epistemic-reasoning-risk/) — evidence nodes, Bayesian-network work, and what the store can and cannot support

---

## Deprecated models

Not maintained. Kept because their outputs are on the AI-WQ leaderboard and their docs explain
choices the v2 pipeline inherited.

| model | input · precision | docs |
|---|---|---|
| `FahamuAIFSv1` | ECMWF Open Data · FP32 (A100) | [`FahamuAIFSv1.md`](FahamuAIFSv1/FahamuAIFSv1.md) |
| `fp16FahamuAIFSv1` | ECMWF Open Data · FP16 (G2/L4) | [`fp16FahamuAIFSv1.md`](fp16FahamuAIFSv1/fp16FahamuAIFSv1.md) |
| `era5tFp16FahamuAIFSv1` | CEDA ERA5T · FP16, 10 members, 960 h | [`era5tFp16FahamuAIFSv1.md`](era5tFp16FahamuAIFSv1/era5tFp16FahamuAIFSv1.md) |

They ran on Coiled GPU notebooks against AIFS-ENS **1.0**, with GRIB on GCS rather than a local
Icechunk store. `shared/` still holds the CLIs all of them use — v2 passes `--v2`.

---

## Acknowledgements

1. Hazard modeling, impact estimation, climate storylines for event catalogue on drought and
   flood disasters in the Eastern Africa (E4DRR) project.
   https://icpac-igad.github.io/e4drr/ — United Nations | Complex Risk Analytics Fund (CRAF'd),
   activity 2.3.3: Experiment generative AI for EPS (Ensemble Prediction Systems).
2. The Strengthening Early Warning Systems for Anticipatory Action (SEWAA) Project.
   https://cgan.icpac.net/
