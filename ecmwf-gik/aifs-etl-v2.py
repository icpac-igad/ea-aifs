#!/usr/bin/env python3
"""
ECMWF Parquet to PKL Processor V2 - Optimized

Extracts ALL pressure levels from ECMWF parquet files with parallel S3 fetches.

Optimizations over previous version:
- Single parquet read per member (was 3x)
- Parallel S3 byte-range fetches via ThreadPoolExecutor
- Direct eccodes GRIB decoding (no temp files, no cfgrib overhead)
- Cached obstore S3 connections
- Both timesteps extracted as shape (2, lat, lon) for AIFS

Usage:
    python aifs-etl-v2.py
"""

import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import pickle
import json
import base64
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Configuration
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_CONST = ["lsm", "z", "slor", "sdor"]  # Static forcing from ECMWF Open Data
PARAM_SFC_FC = PARAM_SFC_CONST  # alias for verification
PARAM_SOIL = ["sot"]
SOIL_LEVELS = [1, 2]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

date_str = '20260319'
run = '00'
OUTPUT_DIR = f"{date_str}_{run}_ecmwf_pkl_par"
SAVE_STATES = True
ALL_MEMBERS = [-1] + list(range(1, 51))  # 51 total

# Parallel fetch config
MAX_WORKERS = 16

# GCS config — upload each pkl then delete local to save disk
GCS_BUCKET = "aifs-aiquest-us-20251127"
GCS_SERVICE_ACCOUNT_KEY = "/scratch/notebook/ea-aifs/coiled-data.json"
UPLOAD_TO_GCS = True
CLEANUP_LOCAL = True


# --- S3 fetch layer (cached stores, parallel-ready) ---

_obstore_cache = {}

def _get_obstore(bucket):
    """Get or create a cached obstore S3 store."""
    if bucket not in _obstore_cache:
        from obstore.store import from_url
        regions = {'ecmwf-forecasts': 'eu-central-1'}
        region = regions.get(bucket, 'eu-central-1')
        _obstore_cache[bucket] = from_url(f"s3://{bucket}", region=region, skip_signature=True)
    return _obstore_cache[bucket]


def fetch_s3_chunk(url, offset, length):
    """Fetch a byte range from S3 using obstore (fast) with fsspec fallback."""
    try:
        import obstore as obs
        parts = url[5:].split('/', 1)  # strip s3://
        store = _get_obstore(parts[0])
        result = obs.get_range(store, parts[1], start=offset, end=offset + length)
        return bytes(result)
    except Exception:
        import fsspec
        fs = fsspec.filesystem('s3', anon=True)
        with fs.open(url[5:] if url.startswith('s3://') else url, 'rb') as f:
            f.seek(offset)
            return f.read(length)


def decode_grib_bytes(data):
    """Decode GRIB2 bytes directly with eccodes (no temp files)."""
    import eccodes
    msgid = eccodes.codes_new_from_message(data)
    try:
        ni = eccodes.codes_get(msgid, 'Ni')
        nj = eccodes.codes_get(msgid, 'Nj')
        values = eccodes.codes_get_array(msgid, 'values').astype(np.float32)
        return values.reshape(nj, ni)
    finally:
        eccodes.codes_release(msgid)


# --- Parquet reading (single read, reused) ---

def read_parquet_to_refs(parquet_path):
    """Read parquet file once and return zarr reference dict."""
    df = pd.read_parquet(parquet_path)
    zstore = {}
    for _, row in df.iterrows():
        key = row['key']
        value = row['value']
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                value = json.loads(value)
            except Exception:
                pass
        zstore[key] = value
    zstore.pop('version', None)
    return zstore


def get_variable_path_mapping():
    return {
        '10u': 'u10/instant/heightAboveGround/u10',
        '10v': 'v10/instant/heightAboveGround/v10',
        '2t': 't2m/instant/heightAboveGround/t2m',
        '2d': 'd2m/instant/heightAboveGround/d2m',
        'msl': 'msl/instant/meanSea/msl',
        'sp': 'sp/instant/surface/sp',
        'skt': 'skt/instant/surface/skt',
        'tcw': 'tcw/instant/entireAtmosphere/tcw',
        'lsm': 'lsm/instant/surface/lsm',
        'z': 'z/instant/surface/z',
        'slor': 'slor/instant/surface/slor',
        'sdor': 'sdor/instant/surface/sdor',
        'gh': 'gh/instant/isobaricInhPa/gh',
        't': 't/instant/isobaricInhPa/t',
        'u': 'u/instant/isobaricInhPa/u',
        'v': 'v/instant/isobaricInhPa/v',
        'w': 'w/instant/isobaricInhPa/w',
        'q': 'q/instant/isobaricInhPa/q',
        # Soil temperature
        'sot': 'sot/instant/soilLayer/sot',
    }


# --- Parallel variable extraction ---

def _collect_s3_refs(zstore, variable_path):
    """Collect S3 chunk references, filtering out-of-bounds indices."""
    # Get shape to filter invalid chunks
    zarray_key = f"{variable_path}/.zarray"
    shape = None
    chunks = None
    if zarray_key in zstore:
        meta = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]
        shape = tuple(meta['shape'])
        chunks = tuple(meta['chunks'])

    refs = []
    for key in sorted(zstore.keys()):
        if key.startswith(variable_path + "/") and not key.endswith(('.zarray', '.zattrs', '.zgroup')):
            val = zstore[key]
            if isinstance(val, list) and len(val) >= 3:
                url, offset, length = val[0], val[1], val[2]
                if isinstance(url, str) and 's3://' in url:
                    chunk_idx = key.replace(variable_path + "/", "")
                    # Filter out-of-bounds chunks (from imperfect member extraction)
                    if shape and chunks:
                        indices = [int(x) for x in chunk_idx.split('.')]
                        in_bounds = True
                        for dim_i, (idx, cs, ds) in enumerate(zip(indices, chunks, shape)):
                            if idx * cs >= ds:
                                in_bounds = False
                                break
                        if not in_bounds:
                            continue
                    refs.append((chunk_idx, url, offset, length))
    return refs


def _fetch_and_decode(args):
    """Fetch one S3 chunk and decode GRIB. Thread-safe."""
    chunk_idx, url, offset, length = args
    data = fetch_s3_chunk(url, offset, length)
    if data[:4] == b'GRIB':
        return chunk_idx, decode_grib_bytes(data)
    return chunk_idx, np.frombuffer(data, dtype=np.float32)


def extract_variable_parallel(zstore, variable_path):
    """Extract a variable with parallel S3 fetches and GRIB decoding."""
    zarray_key = f"{variable_path}/.zarray"
    if zarray_key not in zstore:
        return None

    metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]
    shape = tuple(metadata['shape'])
    chunks = tuple(metadata['chunks'])

    s3_refs = _collect_s3_refs(zstore, variable_path)
    if not s3_refs:
        return None

    # Parallel fetch + decode
    chunks_data = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_and_decode, ref): ref[0] for ref in s3_refs}
        for future in as_completed(futures):
            try:
                chunk_idx, arr = future.result()
                chunks_data[chunk_idx] = arr
            except Exception as e:
                print(f"      Chunk fetch failed: {e}")

    if not chunks_data:
        return None

    # Reassemble array
    if len(chunks_data) == 1:
        arr = list(chunks_data.values())[0]
        if isinstance(arr, np.ndarray) and arr.size == np.prod(shape):
            return arr.reshape(shape)
        return arr

    # Multi-chunk reassembly
    dtype = list(chunks_data.values())[0].dtype if isinstance(list(chunks_data.values())[0], np.ndarray) else np.float32
    array = np.zeros(shape, dtype=dtype)

    for chunk_idx_str, chunk_arr in chunks_data.items():
        indices = tuple(int(x) for x in chunk_idx_str.split('.'))
        if isinstance(chunk_arr, np.ndarray) and chunk_arr.ndim == 2 and len(shape) >= 4:
            # GRIB2 2D -> fill into ND array
            if len(shape) == 4:
                array[indices[0], indices[1], :, :] = chunk_arr
            elif len(shape) == 5:
                array[indices[0], indices[1], indices[2] if len(indices) > 2 else 0, :, :] = chunk_arr
        else:
            if not isinstance(chunk_arr, np.ndarray):
                chunk_arr = np.frombuffer(chunk_arr, dtype=dtype)
            slices = []
            for idx, cs, ds in zip(indices, chunks, shape):
                start = idx * cs
                end = min(start + cs, ds)
                slices.append(slice(start, end))
            chunk_shape = tuple(s.stop - s.start for s in slices)
            if chunk_arr.size == np.prod(chunk_shape):
                chunk_arr = chunk_arr.reshape(chunk_shape)
            array[tuple(slices)] = chunk_arr

    return array


def extract_pressure_level_coordinates(zstore, base_path):
    """Extract pressure level coordinate values IN ARRAY ORDER (not sorted)."""
    levels = []
    coord_base = f"{base_path}/isobaricInhPa"
    zarray_key = f"{coord_base}/.zarray"
    if zarray_key in zstore:
        try:
            metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]
            num_levels = metadata.get('shape', [0])[0]
            # Try reading individual chunks first
            for i in range(num_levels):
                coord_path = f"{coord_base}/{i}"
                if coord_path in zstore:
                    val = zstore[coord_path]
                    if isinstance(val, str) and val.startswith('base64:'):
                        decoded = base64.b64decode(val[7:])
                        if len(decoded) == 8:
                            # Single value per chunk
                            levels.append(int(struct.unpack('<d', decoded)[0]))
                        elif len(decoded) == num_levels * 8:
                            # All values packed in one chunk (common case)
                            all_vals = struct.unpack(f'<{num_levels}d', decoded)
                            levels = [int(v) for v in all_vals]
                            break
        except Exception:
            pass
    # Return in ARRAY ORDER — do NOT sort, as this matches the dimension index
    return levels


def get_constant_fields():
    """Get static forcing fields (lsm, z, slor, sdor) from ECMWF Open Data.

    Same approach as ceda_era5t_pkl_input_aifsens.py — these are time-invariant
    constants downloaded once and shared across all members. Not available in
    the ensemble GRIB parquets, so fetched separately.

    Returns dict of {param: np.array shape (2, 721, 1440)} at 0.25 deg.
    Both timesteps contain identical values (constants don't change).
    """
    import earthkit.data as ekd

    print("  Fetching constant fields (z, slor, sdor, lsm) from ECMWF Open Data...")
    data = ekd.from_source("ecmwf-open-data", param=PARAM_SFC_CONST)

    fields = {}
    for f in data:
        values = f.to_numpy()
        assert values.shape == (721, 1440), f"Unexpected shape: {values.shape}"
        # ECMWF Open Data is -180 to 180, shift to 0-360
        values = np.roll(values, -values.shape[1] // 2, axis=1).astype(np.float32)
        # Stack same values for both timesteps (constants don't change)
        fields[f.metadata("param")] = np.stack([values, values])
        print(f"    {f.metadata('param')}: shape={fields[f.metadata('param')].shape}, "
              f"range=[{values.min():.4f}, {values.max():.4f}]")

    return fields


def get_soil_fields(date, member_num):
    """Get soil temperature (stl1, stl2) from ECMWF Open Data for a specific member.

    Soil temps are per-member ensemble fields (not constants), fetched from
    ECMWF Open Data at two timesteps (t-6h and t). Same approach as
    ecmwf_opendata_pkl_input_aifsens.py.

    Returns dict {'stl1': array(2, 542080), 'stl2': array(2, 542080)} on N320 grid.
    """
    import earthkit.data as ekd
    import earthkit.regrid as ekr
    from collections import defaultdict

    fields = defaultdict(list)
    # Control member uses number=0 in ECMWF Open Data
    number = 0 if member_num == -1 else member_num

    for d in [date - datetime.timedelta(hours=6), date]:
        data = ekd.from_source("ecmwf-open-data", date=d, param=PARAM_SOIL,
                               levelist=SOIL_LEVELS, number=[number], stream='enfo')
        for f in data:
            values = f.to_numpy()
            assert values.shape == (721, 1440), f"Unexpected shape: {values.shape}"
            values = np.roll(values, -values.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            name = f"{f.metadata('param')}_{f.metadata('levelist')}"
            fields[name].append(values)

    result = {}
    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2'}
    for k, v in fields.items():
        new_name = mapping.get(k, k)
        result[new_name] = np.stack(v)

    return result


def extract_all_fields(zstore, const_fields=None):
    """Extract all AIFS fields from a single pre-loaded zstore. Returns fields dict.

    const_fields: pre-fetched constant fields (lsm, z, slor, sdor) to merge in.
    """
    var_paths = get_variable_path_mapping()
    fields = {}
    # Fetch dynamic params + soil from parquet; constants come from const_fields
    all_params = PARAM_SFC + ['sot'] + PARAM_PL

    for p in all_params:
        if p not in var_paths:
            continue
        base_path = var_paths[p]
        is_pressure = p in PARAM_PL

        t0 = time.time()
        array = extract_variable_parallel(zstore, base_path)
        fetch_time = time.time() - t0

        if array is None:
            print(f"    {p}: not found in parquet")
            continue

        if is_pressure:
            actual_levels = extract_pressure_level_coordinates(zstore, base_path.rsplit('/', 1)[0])
            level_mapping = actual_levels if actual_levels and len(actual_levels) == array.shape[2 if array.ndim == 5 else (1 if array.ndim == 4 else 0)] else LEVELS

            if array.ndim == 5:
                # (time=2, step=1, level=13, lat, lon)
                for li, lv in enumerate(level_mapping):
                    if lv in LEVELS:
                        fields[f"{p}_{lv}"] = array[:, 0, li, :, :]
            elif array.ndim == 4:
                for li, lv in enumerate(level_mapping):
                    if lv in LEVELS:
                        fields[f"{p}_{lv}"] = array[:, li, :, :]
            elif array.ndim == 3:
                for li, lv in enumerate(level_mapping):
                    if lv in LEVELS:
                        fields[f"{p}_{lv}"] = array[li, :, :]

            print(f"    {p}: {len(LEVELS)} levels, shape {list(fields.values())[-1].shape} ({fetch_time:.1f}s)")
        else:
            # Surface: (time=2, step=1, lat, lon) -> (2, lat, lon)
            if array.ndim == 4:
                fields[p] = array[:, 0, :, :]
            elif array.ndim == 3:
                fields[p] = array
            elif array.ndim == 2:
                fields[p] = array
            else:
                fields[p] = array.reshape(array.shape[-2:])
            print(f"    {p}: shape {fields[p].shape} ({fetch_time:.1f}s)")

    # Convert gh -> z (geopotential)
    for level in LEVELS:
        gh_key = f"gh_{level}"
        if gh_key in fields:
            fields[f"z_{level}"] = fields.pop(gh_key) * 9.80665

    # Extract soil temperature from parquet if available
    # The parquet has sot as a single layer; use for both stl1 and stl2
    if 'sot' in fields:
        sot_data = fields.pop('sot')
        fields['stl1'] = sot_data
        fields['stl2'] = sot_data.copy()
        print(f"    sot -> stl1, stl2: shape {sot_data.shape}")

    # Merge constant forcing fields (lsm, z, slor, sdor) from ECMWF Open Data
    if const_fields:
        for k, v in const_fields.items():
            fields[k] = v

    # Regrid all fields from 0.25° (721x1440) to N320 for AIFS
    import earthkit.regrid as ekr
    regridded = {}
    for fname, fdata in fields.items():
        if fdata.ndim == 2 and fdata.shape == (721, 1440):
            regridded[fname] = ekr.interpolate(fdata, {"grid": (0.25, 0.25)}, {"grid": "N320"})
        elif fdata.ndim == 2:
            regridded[fname] = fdata  # already regridded (e.g. from constants)
        else:
            # Multi-timestep: regrid each timestep
            steps = []
            for t in range(fdata.shape[0]):
                if fdata[t].shape == (721, 1440):
                    steps.append(ekr.interpolate(fdata[t], {"grid": (0.25, 0.25)}, {"grid": "N320"}))
                else:
                    steps.append(fdata[t])
            regridded[fname] = np.stack(steps)
    print(f"    Regridded {len(regridded)} fields to N320")

    return regridded


def create_input_state_from_parquet(parquet_path, member, zstore=None, const_fields=None):
    """Create input state from parquet. Accepts pre-loaded zstore to avoid re-reading."""
    member_label = "control" if member == -1 else f"ens_{member:02d}"
    print(f"\n  [{member_label}] Extracting fields...")
    start_time = time.time()

    if zstore is None:
        zstore = read_parquet_to_refs(parquet_path)

    fields = extract_all_fields(zstore, const_fields=const_fields)

    # Extract date from path
    path_parts = Path(parquet_path).parts
    date = datetime.datetime.now()
    for part in path_parts:
        if 'ecmwf_' in part and '_efficient' in part:
            dp = part.replace('ecmwf_', '').replace('_efficient', '').split('_')
            if len(dp) >= 2:
                date = datetime.datetime.strptime(f"{dp[0]}_{dp[1]}", "%Y%m%d_%H")
                break

    elapsed = time.time() - start_time
    print(f"  [{member_label}] {len(fields)} fields in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    return dict(date=date, fields=fields)


def verify_input_state(input_state, member):
    """Verify the input state has all required fields."""
    fields = input_state['fields']
    expected_surface = PARAM_SFC + PARAM_SFC_FC + ['stl1', 'stl2']
    expected_pressure = []
    for param in PARAM_PL:
        pname = 'z' if param == 'gh' else param
        for level in LEVELS:
            expected_pressure.append(f"{pname}_{level}")

    expected_total = expected_surface + expected_pressure
    available = [f for f in expected_total if f in fields]
    missing = [f for f in expected_total if f not in fields]

    if missing:
        missing_sfc = [f for f in missing if '_' not in f]
        missing_pl = [f for f in missing if '_' in f]
        label = "control" if member == -1 else f"ens_{member:02d}"
        parts = []
        if missing_sfc:
            parts.append(f"sfc:{missing_sfc}")
        if missing_pl:
            parts.append(f"pl:{len(missing_pl)}")
        print(f"  [{label}] {len(available)}/{len(expected_total)} fields (missing: {', '.join(parts)})")
        return len(missing_pl) == 0  # OK if only missing static sfc fields
    else:
        label = "control" if member == -1 else f"ens_{member:02d}"
        print(f"  [{label}] {len(available)}/{len(expected_total)} fields - complete")
        return True


def upload_to_gcs(local_path, bucket_name, blob_name, sa_key):
    """Upload file to GCS and return success."""
    from google.cloud import storage
    try:
        client = storage.Client.from_service_account_json(sa_key)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        print(f"  Uploaded to gs://{bucket_name}/{blob_name}")
        return True
    except Exception as e:
        print(f"  GCS upload failed: {e}")
        return False


def main():
    """Process all 51 ensemble members: parquet -> PKL -> GCS."""
    base_dir = f"ecmwf_{date_str}_{run}_efficient/members"
    datestr = f"{date_str}_{run}00"

    print("="*70)
    print(f"ECMWF PARQUET->PKL V2 (Optimized) | {date_str} {run}z | {len(ALL_MEMBERS)} members")
    print(f"Parallel S3 fetches: {MAX_WORKERS} workers | Output: {OUTPUT_DIR}/ -> GCS")
    print("="*70)

    if SAVE_STATES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check GCS credentials
    gcs_enabled = UPLOAD_TO_GCS and Path(GCS_SERVICE_ACCOUNT_KEY).exists()
    if UPLOAD_TO_GCS and not gcs_enabled:
        print(f"  GCS key not found: {GCS_SERVICE_ACCOUNT_KEY}, saving locally only")
    if gcs_enabled:
        print(f"  GCS: gs://{GCS_BUCKET}/{datestr}/input/")
        if CLEANUP_LOCAL:
            print(f"  Local cleanup enabled — pkl deleted after GCS upload")

    # Fetch constant fields once (lsm, z, slor, sdor) from ECMWF Open Data
    # Same approach as ceda_era5t_pkl_input_aifsens.py
    try:
        const_fields = get_constant_fields()
        print(f"  Constant fields: {list(const_fields.keys())}")
    except Exception as e:
        print(f"  Warning: Could not fetch constant fields: {e}")
        print(f"  Continuing without z, slor, sdor (will have 87 fields instead of 90)")
        const_fields = None

    successful = []
    failed = []
    total_start = time.time()

    for i, member in enumerate(ALL_MEMBERS):
        member_name = "control" if member == -1 else f"ens_{member:02d}"
        member_num = 0 if member == -1 else member
        parquet_file = f"{base_dir}/{member_name}/{member_name}.parquet"

        if not Path(parquet_file).exists():
            print(f"\n  [{member_name}] parquet not found, skipping")
            failed.append(member_name)
            continue

        try:
            # Single parquet read
            zstore = read_parquet_to_refs(parquet_file)
            input_state = create_input_state_from_parquet(parquet_file, member=member, zstore=zstore, const_fields=const_fields)
            is_valid = verify_input_state(input_state, member=member)

            if SAVE_STATES:
                output_file = f"{OUTPUT_DIR}/input_state_member_{member_num:03d}.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(input_state, f)
                file_size = Path(output_file).stat().st_size / (1024 * 1024)
                print(f"  [{member_name}] Saved {output_file} ({file_size:.0f} MB)")

                # Upload to GCS and cleanup local
                if gcs_enabled:
                    gcs_blob = f"{datestr}/input/input_state_member_{member_num:03d}.pkl"
                    uploaded = upload_to_gcs(output_file, GCS_BUCKET, gcs_blob, GCS_SERVICE_ACCOUNT_KEY)
                    if uploaded and CLEANUP_LOCAL:
                        os.remove(output_file)
                        print(f"  [{member_name}] Local file removed")

            successful.append(member_name)

            # ETA
            elapsed_total = time.time() - total_start
            avg_per_member = elapsed_total / (i + 1)
            remaining = avg_per_member * (len(ALL_MEMBERS) - i - 1)
            print(f"  Progress: {i+1}/{len(ALL_MEMBERS)} | ETA: {remaining/60:.0f} min")

        except Exception as e:
            print(f"\n  [{member_name}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append(member_name)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"DONE | {len(successful)}/{len(ALL_MEMBERS)} successful | {total_time/60:.1f} min total")
    if failed:
        print(f"Failed: {failed}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
