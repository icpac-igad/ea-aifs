#!/usr/bin/env python3
"""
AIFS ENS v2.0 — input-state (pkl) creation from ECMWF Open Data (CPU only).

Builds per-member ``input_state`` pickle files for the **AIFS ENS v2.0** model
(checkpoint ``ecmwf/aifs-ens-2.0``) and uploads them to GCS, mirroring the data
preparation in the official notebook ``run_AIFS_ENS_v2.0.ipynb``.

What is different from the v1 prep (`shared/ecmwf_opendata_pkl_input_aifsens.py`):
  * Surface adds ``sd`` (snow depth).
  * Soil adds ``vsw`` (soil moisture) → ``swvl1/swvl2`` alongside ``stl1/stl2``.
  * A whole new **wave** group (11 params from the ``waef``/``wave`` stream),
    with ``mwd`` decomposed into ``cos_mwd``/``sin_mwd``.
  * Pressure levels gain **level 10 hPa** (14 levels), and ``q_10`` is dropped.
  * A land–sea mask sets ``sd``, ``swvl1``, ``swvl2`` to NaN over sea.
  * ``gh → z`` (×g) is kept, same as v1.

No GPU and no model checkpoint are required — this is data retrieval + transforms
only. The resulting pkl (~112 fields, 2 timesteps each) feeds the v2 GPU runner.
"""

import argparse
import datetime
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
from google.cloud import storage
from ecmwf.opendata import Client as OpendataClient

# ---------------------------------------------------------------------------
# v2.0 parameter lists (from run_AIFS_ENS_v2.0.ipynb)
# ---------------------------------------------------------------------------
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "sd"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]              # time-invariant constants
PARAM_SOIL = ["vsw", "sot"]                               # → swvl1/2, stl1/2
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721",
              "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
SOIL_LEVELS = [1, 2]

SOIL_RENAME = {"sot_1": "stl1", "sot_2": "stl2",
               "vsw_1": "swvl1", "vsw_2": "swvl2"}

# GCS configuration (same bucket/key conventions as the v1 prep)
GCS_BUCKET = "aifs-aiquest-us-20251127"
GCS_SERVICE_ACCOUNT_KEY = "coiled-data.json"
GCS_SUBPATH = "input_v2"          # keep v2 pkls separate from v1 ("input/")
G = 9.80665                       # standard gravity, for gh → z


def get_open_data(date, param, levelist=[], number=None, constant=False,
                  source="ecmwf", **kwargs):
    """Retrieve + N320-interpolate ECMWF Open Data for a (param, level) group.

    Returns a dict ``name -> ndarray`` stacked over [date-6h, date] (2 timesteps).

    ``constant=True``: time-invariant fields (lsm, z, slor, sdor). The deterministic
    stream only runs 00z/12z, so a ``date-6h`` step landing on 06z/18z 404s — fetch
    once at ``date`` and replicate to the 2-timestep shape used by prognostic fields.

    ``number`` selects an ensemble perturbation; ``stream`` (via kwargs) picks the
    archive (``enfo`` for atmosphere, ``waef`` for wave ensemble).
    """
    fields = defaultdict(list)
    dates = [date] if constant else [date - datetime.timedelta(hours=6), date]
    for d in dates:
        if number is None:
            data = ekd.from_source("ecmwf-open-data", date=d, param=param,
                                   levelist=levelist, source=source, **kwargs)
        else:
            kwargs.setdefault("stream", "enfo")
            data = ekd.from_source("ecmwf-open-data", date=d, param=param,
                                   levelist=levelist, number=[number],
                                   source=source, **kwargs)

        for f in data:
            # 50r1 added pressure-level `z` to the deterministic stream; when no
            # level list was requested keep only single-level fields, else the
            # constant `z` (orography) collapses into a (2*Nlevels,...) array.
            if not levelist and f.metadata("levtype") == "pl":
                continue
            # Open data is -180..180; shift to 0..360, then interpolate 0.25 -> N320
            assert f.to_numpy().shape == (721, 1440), \
                f"unexpected grid {f.to_numpy().shape} for {f.metadata('param')}"
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist \
                else f.metadata("param")
            fields[name].append(values)
            if constant:                       # fetched once -> replicate 2nd step
                fields[name].append(values)

    for name, values in fields.items():
        fields[name] = np.stack(values)
    return fields


def create_input_state(date, number, source="ecmwf"):
    """Build the v2.0 input_state dict for one ensemble member."""
    print(f"\nCreating v2.0 input state for ensemble member {number}")
    t0 = time.time()
    fields = {}

    print("  surface fields...")
    fields.update(get_open_data(date, param=PARAM_SFC, number=number,
                                levtype="sfc", source=source))

    print("  constant surface fields...")
    fields.update(get_open_data(date, param=PARAM_SFC_FC, constant=True,
                                levtype="sfc", source=source))

    print("  wave fields...")
    wave_stream = "wave" if not number else "waef"
    fields.update(get_open_data(date, param=PARAM_WAVE, number=number,
                                stream=wave_stream, source=source))

    print("  soil fields...")
    soil = get_open_data(date, param=PARAM_SOIL, levelist=SOIL_LEVELS,
                         number=number, source=source)
    for k, v in soil.items():
        fields[SOIL_RENAME[k]] = v

    print("  pressure-level fields...")
    fields.update(get_open_data(date, param=PARAM_PL, levelist=LEVELS,
                                number=number, source=source))

    # ----- v2.0 data transformations -----
    # mwd -> cos/sin components
    mwd_rad = np.deg2rad(fields.pop("mwd"))
    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad)

    # drop unused 10 hPa specific humidity
    fields.pop("q_10", None)

    # land-sea mask: NaN over sea for snow depth + soil moisture.
    # Derive the mask from the (already N320) lsm field rather than a side file.
    mask = np.equal(fields["lsm"][0], 0)
    for f in ("sd", "swvl1", "swvl2"):
        fields[f][:, mask] = np.nan

    # geopotential height -> geopotential
    for level in LEVELS:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * G

    input_state = dict(date=date, fields=fields)
    print(f"  done in {time.time()-t0:.1f}s — {len(fields)} fields, "
          f"shape {next(iter(fields.values())).shape}")
    return input_state


def expected_field_names():
    """The full set of v2.0 field names, for verification."""
    names = list(PARAM_SFC) + list(PARAM_SFC_FC) + ["stl1", "stl2", "swvl1", "swvl2"]
    names += [w for w in PARAM_WAVE if w != "mwd"] + ["cos_mwd", "sin_mwd"]
    for p in PARAM_PL:
        out = "z" if p == "gh" else p
        for lev in LEVELS:
            if out == "q" and lev == 10:        # q_10 dropped
                continue
            names.append(f"{out}_{lev}")
    return names


def verify_input_state(input_state, member):
    fields = input_state["fields"]
    expected = expected_field_names()
    missing = [f for f in expected if f not in fields]
    extra = [f for f in fields if f not in expected]
    print(f"  verify member {member}: {len(fields)} fields "
          f"(expected {len(expected)})")
    if missing:
        print(f"    ⚠️  missing: {missing}")
    if extra:
        print(f"    ⚠️  extra:   {extra}")
    ok = not missing and not extra
    print("    ✅ all fields present" if ok else "    ❌ field mismatch")
    return ok


def upload_to_gcs(local_path, bucket, blob_name, key):
    try:
        client = storage.Client.from_service_account_json(key)
        client.bucket(bucket).blob(blob_name).upload_from_filename(local_path)
        print(f"    ✅ uploaded gs://{bucket}/{blob_name}")
        return True
    except Exception as e:
        print(f"    ❌ GCS upload failed: {e}")
        return False


def parse_members(spec):
    """Parse '1-50', '1,5,10' or '3' into a sorted list of ints."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description="AIFS ENS v2.0 input-state pkl creator")
    ap.add_argument("--date", help="init date YYYYMMDD or YYYYMMDD_HHMM "
                                    "(default: latest open-data)")
    ap.add_argument("--members", default="1-50",
                    help="ensemble members, e.g. '1-50', '1,5,10', '3' (default 1-50)")
    ap.add_argument("--source", default="ecmwf",
                    choices=["ecmwf", "azure", "aws", "google"],
                    help="ECMWF open-data mirror (default ecmwf)")
    ap.add_argument("--gcs-subpath", default=GCS_SUBPATH,
                    help=f"GCS subfolder under the date (default {GCS_SUBPATH})")
    ap.add_argument("--no-upload", action="store_true", help="skip GCS upload")
    ap.add_argument("--keep-local", action="store_true",
                    help="keep local pkl files after upload")
    ap.add_argument("--out-dir", default="input_states_v2",
                    help="local output dir (default input_states_v2)")
    args = ap.parse_args()

    if args.date:
        d = args.date.replace("_", "")
        DATE = datetime.datetime.strptime(d[:8] + (d[8:12] or "0000"), "%Y%m%d%H%M")
    else:
        DATE = OpendataClient(args.source).latest()
    datestr = DATE.strftime("%Y%m%d_%H%M")
    members = parse_members(args.members)
    upload = not args.no_upload
    if upload and not os.path.exists(GCS_SERVICE_ACCOUNT_KEY):
        print(f"⚠️  {GCS_SERVICE_ACCOUNT_KEY} not found — disabling GCS upload")
        upload = False

    ekd.config.set({"cache-policy": "user"})
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"AIFS ENS v2.0 input prep | init {DATE} | members {members[0]}-{members[-1]}"
          f" | source {args.source} | upload={upload}")

    ok, fail, times = [], [], []
    for i, m in enumerate(members):
        print(f"\n{'='*60}\nMember {m} ({i+1}/{len(members)})\n{'='*60}")
        try:
            t0 = time.time()
            state = create_input_state(DATE, m, source=args.source)
            if not verify_input_state(state, m):
                fail.append(m)
                continue
            fname = os.path.join(args.out_dir, f"input_state_member_{m:03d}.pkl")
            with open(fname, "wb") as fh:
                pickle.dump(state, fh)
            print(f"  saved {fname}")
            if upload:
                blob = f"{datestr}/{args.gcs_subpath}/input_state_member_{m:03d}.pkl"
                if upload_to_gcs(fname, GCS_BUCKET, blob, GCS_SERVICE_ACCOUNT_KEY) \
                        and not args.keep_local:
                    os.remove(fname)
            ok.append(m)
            times.append(time.time() - t0)
            avg = sum(times) / len(times)
            print(f"  {times[-1]:.1f}s | ETA {(len(members)-i-1)*avg/60:.1f} min")
        except Exception as e:
            print(f"  ❌ member {m} failed: {e}")
            fail.append(m)

    print(f"\n{'='*60}\nSUMMARY: {len(ok)}/{len(members)} ok"
          + (f", failed {fail}" if fail else ""))
    if times:
        print(f"  total {sum(times)/60:.1f} min, avg {sum(times)/len(times):.1f}s/member")


if __name__ == "__main__":
    main()
