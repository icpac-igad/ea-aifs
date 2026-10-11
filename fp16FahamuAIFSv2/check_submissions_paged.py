#!/usr/bin/env python3
"""Verify AI-WQ submissions, following ECBox pagination.

`AI_WQ_package.forecast_submission.AI_WQ_check_submission` calls
`content_manager.list(remote_path=...)` **once** and ignores the
`continuation_token` the response carries. ECBox pages at 100 entries, and a
competition date holds hundreds of files across all teams, so the official check
reports a **false ABSENT** for any filename sorting after the 100th entry.

Measured on `20261008`: the first page ended at
`mslp_20261008_p2_NMCBridge_multical.nc`, so every `pr_*` and `tas_*` file was
invisible and reported missing — while 3c had in fact submitted all six.

The asymmetry matters for which results you can trust:

    MJO_*, TS_*      uppercase, sort EARLY  -> reliably on page 1
    mslp_*, pr_*, tas_*   lowercase, sort LATE -> may be on page 2+

So an ABSENT verdict from the official checker is only trustworthy for TS and
MJO. For the gridded variables it means nothing without pagination.

Usage:
    python check_submissions_paged.py --date 20261008
"""
from __future__ import annotations

import argparse
import os


def load_env(path=".env"):
    for ln in open(path):
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.strip().split("=", 1)
            os.environ.setdefault(k, v)


def list_all(cm, remote_dir):
    """Every entry in remote_dir, following continuation tokens."""
    names, token, pages = [], None, 0
    while True:
        kw = {"remote_path": remote_dir}
        if token:
            kw["continuation_token"] = token
        resp = cm.list(**kw)
        names += [os.path.basename(f["path"]) for f in resp["files"]]
        pages += 1
        token = resp.get("continuation_token")
        if not token:
            break
        if pages > 200:                      # refuse to loop forever
            raise RuntimeError("more than 200 pages; aborting")
    return names, pages


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args()
    load_env(args.env_file)
    team = os.environ["AIWQ_TEAM_NAME"]
    model = os.environ.get("AIWQ_MODEL_NAME_V2") or os.environ["AIWQ_MODEL_NAME_FP16"]
    pw = os.environ.get("AIWQ_ECBOX_TOKEN") or os.environ.get("ecbox")

    import AI_WQ_package.forecast_submission as FS
    site = FS.Site.from_space_and_name(space="ecbox", name="AI_Weather_Quest")
    cm = site.get_content_manager(authenticator=FS.Authenticator.from_token(token=pw))
    names, pages = list_all(cm, f"forecast_submissions/{args.date}")
    print(f"{args.date}: {len(names)} files across {pages} page(s)")

    expected = [(v, p) for v in ("tas", "mslp", "pr") for p in ("1", "2")] \
               + [("TS", "1"), ("TS", "2"), ("MJO", "1")]
    missing = []
    for var, per in expected:
        fn = f"{var}_{args.date}_p{per}_{team}_{model}.nc"
        ok = fn in names
        print(f"  {'OK ' if ok else 'MISSING'}  {var:4s} p{per}")
        if not ok:
            missing.append(fn)
    print(f"\n{len(expected)-len(missing)}/{len(expected)} present")
    if missing:
        print("missing:")
        for m in missing:
            print("   ", m)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
