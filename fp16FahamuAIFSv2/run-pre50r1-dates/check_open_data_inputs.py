#!/usr/bin/env python3
"""Can AIFS-ENS v1 / v2 be initialised for a given date from ECMWF open data?

Reads the open-data GRIB ``.index`` files for a cycle and checks every field each model
version needs. Use it before committing to a run, and to find the earliest runnable date
after any future IFS cycle change.

The answer for v2 is a hard floor at **2026-05-13**: the six period-band significant wave
heights (``h1012`` .. ``h2530``), ``cdww`` and ``wmb`` are fields introduced by IFS Cycle
50r1, which went operational with the 06 UTC run on 12 May 2026 -- jointly with AIFS v2.
Before that they do not exist in open data, in MARS, or in ERA5. See README.md.

Usage::

    python check_open_data_inputs.py --date 20260212
    python check_open_data_inputs.py --date 20260513 --source aws
    python check_open_data_inputs.py --scan 20260501 20260520     # find the cutover
"""

import argparse
import json
import sys
import urllib.request

GCS = "https://storage.googleapis.com/ecmwf-open-data"
AWS = "https://ecmwf-forecasts.s3.amazonaws.com"

# --- what each model version needs (from the pkl builders) --------------------
V2 = {
    "sfc":  ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "sd"],
    "sol":  ["sot", "vsw"],
    "pl":   ["gh", "t", "u", "v", "w", "q"],
    "levels": [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10],
    "wave": ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530",
             "mwd", "cdww", "mwp", "swh"],
}
V1 = {
    "sfc":  ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"],
    "sol":  ["sot"],
    "pl":   ["gh", "t", "u", "v", "w", "q"],
    "levels": [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    "wave": [],                       # v1 predates the coupled wave inputs
}


def fetch_index(base, date, hh, stream, kind):
    stamp = f"{date}{hh}0000"
    url = f"{base}/{date}/{hh}z/ifs/0p25/{stream}/{stamp}-0h-{stream}-{kind}.index"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return [json.loads(l) for l in r.read().decode().splitlines() if l.strip()]
    except Exception as e:
        return e


def inventory(recs):
    """-> (params by levtype, pressure levels present)"""
    by = {}
    for r in recs:
        by.setdefault(r.get("levtype"), set()).add(r.get("param"))
    lv = sorted({int(r["levelist"]) for r in recs
                 if r.get("levtype") == "pl" and "levelist" in r})
    return by, lv


def check(base, date, hh="00", verbose=True):
    enfo = fetch_index(base, date, hh, "enfo", "ef")
    waef = fetch_index(base, date, hh, "waef", "ef")
    if isinstance(enfo, Exception):
        print(f"  {date}: enfo index unavailable ({enfo})")
        return None
    by, lv = inventory(enfo)
    wave_params = set() if isinstance(waef, Exception) else inventory(waef)[0].get("sfc", set())
    # waef records carry no levtype in some cycles; fall back to every param present
    if isinstance(waef, list) and not wave_params:
        wave_params = {r.get("param") for r in waef}

    results = {}
    for name, spec in (("v2", V2), ("v1", V1)):
        miss = {
            "sfc":  [p for p in spec["sfc"] if p not in by.get("sfc", set())],
            "sol":  [p for p in spec["sol"] if p not in by.get("sol", set())],
            "pl":   [p for p in spec["pl"] if p not in by.get("pl", set())],
            "levels": [l for l in spec["levels"] if l not in lv],
            "wave": [p for p in spec["wave"] if p not in wave_params],
        }
        ok = not any(miss.values())
        results[name] = (ok, miss)
        if verbose:
            print(f"  AIFS {name}: {'RUNNABLE' if ok else 'NOT runnable'}")
            for grp, m in miss.items():
                need = len(spec[grp]) if grp != "levels" else len(spec["levels"])
                if need:
                    print(f"      {grp:7s} {need - len(m)}/{need}"
                          + (f"   missing: {m}" if m else ""))
    return results


def main():
    ap = argparse.ArgumentParser(description="check open-data inputs for AIFS v1/v2")
    ap.add_argument("--date", help="YYYYMMDD")
    ap.add_argument("--time", default="00", help="cycle hour, default 00")
    ap.add_argument("--source", default="gcs", choices=["gcs", "aws"])
    ap.add_argument("--scan", nargs=2, metavar=("START", "END"),
                    help="scan a date range (inclusive) and report v2 runnability")
    args = ap.parse_args()
    base = GCS if args.source == "gcs" else AWS

    if args.scan:
        import datetime
        d = datetime.datetime.strptime(args.scan[0], "%Y%m%d")
        end = datetime.datetime.strptime(args.scan[1], "%Y%m%d")
        print(f"{'date':10s}  v2   v1")
        while d <= end:
            ds = d.strftime("%Y%m%d")
            r = check(base, ds, args.time, verbose=False)
            if r:
                print(f"{ds}  {'YES' if r['v2'][0] else ' no'}  "
                      f"{'YES' if r['v1'][0] else ' no'}")
            d += datetime.timedelta(days=1)
        return 0

    if not args.date:
        ap.error("give --date or --scan")
    print(f"=== ECMWF open data {args.date} {args.time}z ({args.source}) ===")
    r = check(base, args.date, args.time)
    return 0 if r and r["v2"][0] else 1


if __name__ == "__main__":
    sys.exit(main())
