#!/usr/bin/env python3
"""
Reclaim disk from completed AIFS-ENS-2.0 local run cycles.

A full cycle under ``/tank/projects/aifs-run/<date>_0000/`` costs ~630 GB:

    icechunk_v2/    ~584 GB   inference store (native N320)
    input_states/    ~48 GB   50 input pkls (0.97 GB each)
    nc_1p5deg/        ~2 GB   50 regridded 1.5deg NetCDFs
    aiwq/            ~1.7 GB  mostly a temp icechunk store

Only a few MB of that is worth keeping once a cycle is submitted: the quintile
file and the climatology (in ``aiwq/``) plus any plots. This purges the heavy
artifacts and keeps the small ones.

SAFE BY DEFAULT: prints a plan and deletes nothing unless ``--yes`` is passed.
The newest ``--keep-latest`` cycles are protected, as is any cycle a running
process is currently writing to.

Routine use at the start of a new date's run::

    python cleanup_aifs_run.py                 # dry-run: show what would go
    python cleanup_aifs_run.py --yes           # purge all but the newest cycle
    python cleanup_aifs_run.py --date 20260709 --yes    # purge one cycle
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

RUN_ROOT = "/tank/projects/aifs-run"

# Heavy directories purged from a cycle (relative to the cycle dir).
HEAVY_DIRS = (
    "input_states",          # 50 input pkls (~48 GB)
    "nc_1p5deg",             # regridded NetCDFs (~2 GB)
    "ensemble_nc_files",     # temp
)
# Any subdir matching these prefixes is also heavy (stores / GRIB output).
HEAVY_PREFIXES = ("icechunk_v2", "fp16_v2_forecasts")
# Inside aiwq/ only these are purged; the .nc files (quintile + climatology) stay.
AIWQ_PURGE = ("ensemble_icechunk_store", "ensemble_nc_files")

CYCLE_RE = re.compile(r"^\d{8}_\d{4}$")


def du_bytes(path):
    try:
        out = subprocess.run(["du", "-sb", path], capture_output=True, text=True, timeout=120).stdout
        return int(out.split()[0])
    except Exception:
        return 0


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def busy_cycles():
    """Cycle names that a currently-running process references (never purge these)."""
    busy = set()
    try:
        out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return busy
    for line in out.splitlines():
        if "cleanup_aifs_run" in line:
            continue
        for m in re.finditer(r"(\d{8}_\d{4})", line):
            busy.add(m.group(1))
        for m in re.finditer(r"--date\s+(\d{8})", line):
            busy.add(m.group(1) + "_0000")
    return busy


def purge_targets(cycle_dir):
    """List heavy paths inside one cycle directory."""
    targets = []
    for name in sorted(os.listdir(cycle_dir)):
        p = os.path.join(cycle_dir, name)
        if not os.path.isdir(p):
            continue
        if name in HEAVY_DIRS or name.startswith(HEAVY_PREFIXES):
            targets.append(p)
        elif name == "aiwq":
            for sub in sorted(os.listdir(p)):
                if sub in AIWQ_PURGE and os.path.isdir(os.path.join(p, sub)):
                    targets.append(os.path.join(p, sub))
    return targets


def main():
    ap = argparse.ArgumentParser(description="Purge heavy artifacts from completed AIFS run cycles")
    ap.add_argument("--run-root", default=RUN_ROOT)
    ap.add_argument("--date", default=None,
                    help="purge only this cycle (YYYYMMDD or YYYYMMDD_0000)")
    ap.add_argument("--keep-latest", type=int, default=1,
                    help="protect the N most recent cycles (default 1). Ignored with --date.")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete (default is a dry-run plan only)")
    args = ap.parse_args()

    if not os.path.isdir(args.run_root):
        print(f"ERROR: run root not found: {args.run_root}")
        return 1

    cycles = sorted(d for d in os.listdir(args.run_root)
                    if CYCLE_RE.match(d) and os.path.isdir(os.path.join(args.run_root, d)))
    if not cycles:
        print("No cycles found.")
        return 0

    busy = busy_cycles()
    if args.date:
        want = args.date if "_" in args.date else args.date + "_0000"
        selected, protected = [c for c in cycles if c == want], [c for c in cycles if c != want]
        if not selected:
            print(f"ERROR: cycle {want} not found under {args.run_root}")
            return 1
    else:
        keep = set(cycles[-args.keep_latest:]) if args.keep_latest > 0 else set()
        selected = [c for c in cycles if c not in keep]
        protected = [c for c in cycles if c in keep]

    # never purge a cycle something is actively writing to
    blocked = [c for c in selected if c in busy]
    selected = [c for c in selected if c not in busy]

    print(f"Run root : {args.run_root}")
    print(f"Cycles   : {', '.join(cycles)}")
    if protected:
        print(f"Protected (newest {args.keep_latest}): {', '.join(protected)}")
    if blocked:
        print(f"Protected (in use by a running process): {', '.join(blocked)}")
    print(f"Mode     : {'DELETE' if args.yes else 'DRY-RUN (nothing will be removed)'}")
    print("=" * 72)

    total = 0
    plan = []
    for c in selected:
        cdir = os.path.join(args.run_root, c)
        targets = purge_targets(cdir)
        if not targets:
            continue
        print(f"\n{c}:")
        for t in targets:
            sz = du_bytes(t)
            total += sz
            plan.append(t)
            print(f"   purge  {os.path.relpath(t, cdir):<28} {human(sz):>10}")
        kept = [f for f in sorted(os.listdir(os.path.join(cdir, "aiwq")))
                if f.endswith(".nc")] if os.path.isdir(os.path.join(cdir, "aiwq")) else []
        if kept:
            print(f"   keep   aiwq/*.nc                  {len(kept)} files (quintile + climatology)")

    print("\n" + "=" * 72)
    print(f"Reclaimable: {human(total)} across {len(plan)} paths")
    if not plan:
        return 0
    if not args.yes:
        print("Dry-run only. Re-run with --yes to delete.")
        return 0

    freed = 0
    for t in plan:
        try:
            shutil.rmtree(t)
            freed += 1
            print(f"  removed {t}")
        except Exception as e:
            print(f"  FAILED  {t}: {e}")
    print(f"\nDeleted {freed}/{len(plan)} paths, reclaimed ~{human(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
