# dev-test — v2 bring-up / proof-of-concept scripts

Throwaway-but-kept scripts used while bringing up the AIFS-ENS-2.0 (v2) GPU path and
prototyping the Icechunk output. Not part of the production pipeline; handy for
re-validation and as living documentation.

| Script | What it does | Needs |
|---|---|---|
| `smoke_test_v2.py` | Download one member's `input_v2` pkl from GCS, run `ecmwf/aifs-ens-2.0` (FP16) for a short lead time, report VRAM + output GRIB. **No upload.** Used to bring up the v2 env on the L4 (see `../LOAD_TEST_RESULTS.md`). | Ampere+ GPU, v2 software env, `../coiled-data.json`, `HF_HOME`, `CC=/usr/bin/gcc` |
| `icechunk_poc.py` | Write a small `(member,time,values)` ensemble straight to a **GCS-backed Icechunk** store, one transactional commit per member, read it back with xarray, then clean up. Proves the core of Path A (`../ICECHUNK_PATH_A.md`). | `icechunk`, `zarr>=3`, `xarray`, `google-cloud-storage`, `../coiled-data.json` |

```bash
# from the repo root (paths resolve relative to the script)
python fp16FahamuAIFSv2/dev-test/icechunk_poc.py            # CPU, ~seconds, self-cleaning
python fp16FahamuAIFSv2/dev-test/smoke_test_v2.py --lead-time 72   # GPU
```
