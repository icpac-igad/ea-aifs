#!/usr/bin/env python3
"""POC: write an AIFS-ENS-style ensemble straight to a GCS-backed Icechunk store.

Demonstrates the core of Path A (see ../ICECHUNK_PATH_A.md): a `(member, time, values)`
ensemble written to an Icechunk repo on GCS with **one transactional commit per member**,
then read back with xarray. Uses small stand-in dimensions so it runs in seconds; the real
run is M=50, T=161 (=960h/6+1), V=542080 (N320 cells).

By default it writes under a throwaway prefix and deletes the objects at the end
(`--keep` to retain them for inspection).

    python dev-test/icechunk_poc.py                 # run + clean up
    python dev-test/icechunk_poc.py --keep --prefix _icechunk_poc/manual

Requires: icechunk, zarr>=3, xarray, google-cloud-storage and the coiled-data.json
service account (one dir up). Verified against gs://aifs-aiquest-us-20251127.
"""
import argparse
import os

import numpy as np
import zarr
import icechunk
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
DEFAULT_SA = os.path.join(V2, "coiled-data.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", default="aifs-aiquest-us-20251127")
    p.add_argument("--prefix", default="_icechunk_poc/20260625")
    p.add_argument("--service-account", default=DEFAULT_SA)
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--values", type=int, default=2000)
    p.add_argument("--keep", action="store_true", help="do not delete the POC objects")
    args = p.parse_args()
    M, T, V = args.members, args.steps, args.values
    VARS = ["2t", "msl"]

    storage = icechunk.gcs_storage(bucket=args.bucket, prefix=args.prefix,
                                   service_account_file=args.service_account)
    repo = icechunk.Repository.open_or_create(storage)
    print(f"repo on gs://{args.bucket}/{args.prefix}")

    # --- 1. init schema (one commit). Coords carry dimension_names so xarray can read. ---
    s = repo.writable_session("main")
    root = zarr.group(s.store, overwrite=True)
    root.create_array("latitude",  shape=(V,), chunks=(V,), dtype="f4",
                      dimension_names=("values",))[:] = np.linspace(90, -90, V).astype("f4")
    root.create_array("longitude", shape=(V,), chunks=(V,), dtype="f4",
                      dimension_names=("values",))[:] = np.linspace(0, 360, V, endpoint=False).astype("f4")
    root.create_array("time", shape=(T,), chunks=(T,), dtype="i4",
                      dimension_names=("time",))[:] = (np.arange(T) * 6 * 3600).astype("i4")
    root.create_array("member", shape=(M,), chunks=(M,), dtype="i4",
                      dimension_names=("member",))[:] = np.arange(1, M + 1)
    for v in VARS:
        root.create_array(v, shape=(M, T, V), chunks=(1, T, V), dtype="f4",
                          dimension_names=("member", "time", "values"))
    print("commit[init]:", s.commit(f"init schema M={M},T={T},V={V}")[:12])

    # --- 2. one writable session + commit PER MEMBER (the orchestrator's loop) ---
    rng = np.random.default_rng(0)
    for m in range(M):
        s = repo.writable_session("main")
        root = zarr.open_group(s.store, mode="r+")
        for v in VARS:
            base = 288.0 if v == "2t" else 101300.0
            root[v][m, :, :] = (base + rng.standard_normal((T, V)) * 3).astype("f4")
        print(f"commit[member {m+1:03d}]:", s.commit(f"write member {m+1:03d}")[:12])

    # --- 3. history (transactions) ---
    print("\nancestry (newest first):")
    for anc in list(repo.ancestry(branch="main"))[:M + 2]:
        print("  ", anc.id[:12], anc.message)

    # --- 4. read back with xarray ---
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False, zarr_format=3)
    print("\nxarray dataset:\n", ds)
    print("\n2t global mean = %.2f K" % float(ds["2t"].mean()))

    # --- 5. clean up (unless --keep) ---
    if args.keep:
        print(f"\n[keep] objects retained under gs://{args.bucket}/{args.prefix}/")
        return
    from google.cloud import storage as gcs
    c = gcs.Client.from_service_account_json(args.service_account)
    dels = list(c.list_blobs(args.bucket, prefix=args.prefix + "/"))
    for b in dels:
        b.delete()
    print(f"\ncleaned {len(dels)} POC objects under gs://{args.bucket}/{args.prefix}/")


if __name__ == "__main__":
    main()
