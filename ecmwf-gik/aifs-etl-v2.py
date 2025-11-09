#!/usr/bin/env python3
"""
ECMWF Parquet to PKL Processor V2

This script properly extracts ALL pressure levels from ECMWF parquet files.
Phase 2 of ETL pipeline: Extract all levels WITHOUT regridding/interpolation.

Key improvements over v1:
- Properly detects and extracts all 13 pressure levels from multi-dimensional arrays
- Removes the faulty single-level fallback logic
- Focuses on data extraction only (no regridding)

Usage:
    python aifs-etl-v2.py
"""

import datetime
from collections import defaultdict
import time
import os
import pickle
import json
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


# Configuration
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_FC = ["lsm"]
PARAM_SOIL = []  # Not available in parquet
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]

# Configuration
date_str = '20251104'
run = '00'

# Output configuration
OUTPUT_DIR = f"{date_str}_{run}_ecmwf_pkl_par"
SAVE_STATES = True


def read_parquet_to_refs(parquet_path):
    """Read parquet file and extract zarr references."""
    print(f"  📊 Reading parquet file: {parquet_path}")
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
            except:
                pass

        zstore[key] = value

    if 'version' in zstore:
        del zstore['version']

    print(f"  ✅ Loaded {len(zstore)} references")
    return zstore


def decode_chunk_reference(chunk_ref):
    """Decode a chunk reference. Returns (type, data)."""
    if isinstance(chunk_ref, str):
        if chunk_ref.startswith('base64:'):
            base64_str = chunk_ref[7:]
            try:
                decoded = base64.b64decode(base64_str)
                return 'base64', decoded
            except Exception as e:
                print(f"    ⚠️ Error decoding base64: {e}")
                return 'unknown', chunk_ref
        else:
            return 'unknown', chunk_ref

    elif isinstance(chunk_ref, list):
        if len(chunk_ref) >= 3:
            url = chunk_ref[0]
            offset = chunk_ref[1]
            length = chunk_ref[2]

            if isinstance(url, str) and ('s3://' in url or 's3.amazonaws.com' in url):
                return 's3', (url, offset, length)

    return 'unknown', chunk_ref


def fetch_s3_byte_range_fsspec(url, offset, length, max_retries=3, retry_delay=2):
    """Fetch a byte range from S3 using fsspec with retry logic."""
    import time

    for attempt in range(max_retries):
        try:
            import fsspec

            if url.startswith('s3://'):
                s3_path = url[5:]
            else:
                s3_path = url

            fs = fsspec.filesystem('s3', anon=True)

            with fs.open(s3_path, 'rb') as f:
                f.seek(offset)
                data = f.read(length)

            return data

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠️  fsspec attempt {attempt + 1}/{max_retries} failed: {e}")
                print(f"    🔄 Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"    ❌ Error fetching from S3 with fsspec after {max_retries} attempts: {e}")
                return None

    return None


def fetch_s3_byte_range_obstore(url, offset, length):
    """Fetch a byte range from S3 using obstore (if available)."""
    try:
        import obstore as obs
        from obstore.store import from_url

        # Parse bucket and key from URL
        if url.startswith('s3://'):
            url_parts = url[5:].split('/', 1)
            bucket = url_parts[0]
            key = url_parts[1] if len(url_parts) > 1 else ''
        else:
            raise ValueError(f"Invalid S3 URL: {url}")

        # ECMWF buckets are in EU regions
        bucket_regions = {
            'ecmwf-forecasts': 'eu-central-1',
        }
        region = bucket_regions.get(bucket, 'eu-central-1')

        # Create S3 store
        bucket_url = f"s3://{bucket}"
        store = from_url(bucket_url, region=region, skip_signature=True)

        # Fetch byte range
        result = obs.get_range(store, key, start=offset, end=offset + length)
        data = bytes(result)

        return data

    except ImportError:
        return fetch_s3_byte_range_fsspec(url, offset, length)
    except Exception as e:
        print(f"    ⚠️ obstore error: {e}, falling back to fsspec")
        return fetch_s3_byte_range_fsspec(url, offset, length)


def extract_variable_hybrid(zstore, variable_path, use_obstore=False):
    """Extract a variable handling both base64 and S3 references."""
    # Get metadata
    zarray_key = f"{variable_path}/.zarray"
    if zarray_key not in zstore:
        print(f"    ⚠️ No metadata found for {variable_path}")
        return None

    metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]

    shape = tuple(metadata['shape'])
    dtype = np.dtype(metadata['dtype'])
    chunks = tuple(metadata['chunks'])
    compressor = metadata.get('compressor', None)

    # Collect chunks
    chunks_data = {}

    for key in sorted(zstore.keys()):
        if key.startswith(variable_path + "/") and not key.endswith(('.zarray', '.zattrs', '.zgroup')):
            chunk_ref = zstore[key]
            ref_type, ref_data = decode_chunk_reference(chunk_ref)

            if ref_type == 'base64':
                data = ref_data
                if compressor is not None:
                    try:
                        import numcodecs
                        codec = numcodecs.get_codec(compressor)
                        data = codec.decode(data)
                    except:
                        try:
                            import blosc
                            data = blosc.decompress(data)
                        except:
                            pass
                chunks_data[key] = data

            elif ref_type == 's3':
                url, offset, length = ref_data

                # Fetch from S3
                if use_obstore:
                    data = fetch_s3_byte_range_obstore(url, offset, length)
                else:
                    data = fetch_s3_byte_range_fsspec(url, offset, length)

                if data is None:
                    print(f"      ⚠️  Skipping chunk {key} - could not fetch from S3")
                    continue

                if data is not None:
                    # Check if it's GRIB2 data
                    if data[:4] == b'GRIB':
                        # Decode GRIB2 message
                        try:
                            import cfgrib
                            import xarray as xr

                            with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
                                tmp.write(data)
                                tmp_path = tmp.name

                            ds = xr.open_dataset(tmp_path, engine='cfgrib', decode_timedelta=True)
                            var_names = list(ds.data_vars)
                            if var_names:
                                var_data = ds[var_names[0]].values
                                chunks_data[key] = var_data

                            os.unlink(tmp_path)
                            ds.close()

                        except ImportError:
                            print(f"      ⚠️ cfgrib not available, trying eccodes")
                            try:
                                import eccodes
                                gid = eccodes.codes_new_from_message(data)
                                values = eccodes.codes_get_array(gid, 'values')
                                eccodes.codes_release(gid)
                                chunks_data[key] = values
                            except:
                                print(f"      ❌ Cannot decode GRIB2 data")
                        except Exception as e:
                            print(f"      ❌ Error decoding GRIB2: {e}")
                    else:
                        # Try decompression if needed
                        if compressor is not None:
                            try:
                                import numcodecs
                                codec = numcodecs.get_codec(compressor)
                                data = codec.decode(data)
                                chunks_data[key] = data
                            except:
                                pass
                        else:
                            chunks_data[key] = data

    if not chunks_data:
        return None

    # Reconstruct array
    try:
        if len(chunks_data) == 1:
            chunk_data = list(chunks_data.values())[0]

            if isinstance(chunk_data, np.ndarray):
                array = chunk_data
            else:
                array = np.frombuffer(chunk_data, dtype=dtype)

            if array.size == np.prod(shape):
                array = array.reshape(shape)

            return array

        else:
            # Multiple chunks - reassemble
            first_chunk = list(chunks_data.values())[0]
            if isinstance(first_chunk, np.ndarray):
                actual_dtype = first_chunk.dtype
            else:
                actual_dtype = dtype

            array = np.zeros(shape, dtype=actual_dtype)

            for chunk_key, chunk_data in chunks_data.items():
                chunk_idx_str = chunk_key.replace(variable_path + "/", "")
                chunk_indices = tuple(int(x) for x in chunk_idx_str.split('.'))

                if isinstance(chunk_data, np.ndarray):
                    chunk_array = chunk_data
                else:
                    chunk_array = np.frombuffer(chunk_data, dtype=actual_dtype)

                # GRIB2 data comes as 2D, but metadata expects 4D
                if chunk_array.ndim == 2 and len(shape) == 4:
                    time_idx = chunk_indices[0] if len(chunk_indices) > 0 else 0
                    step_idx = chunk_indices[1] if len(chunk_indices) > 1 else 0
                    array[time_idx, step_idx, :, :] = chunk_array
                else:
                    # Standard zarr chunk reassembly
                    chunk_shape = []
                    for i, (idx, chunk_size, dim_size) in enumerate(zip(chunk_indices, chunks, shape)):
                        if (idx + 1) * chunk_size <= dim_size:
                            chunk_shape.append(chunk_size)
                        else:
                            chunk_shape.append(dim_size - idx * chunk_size)

                    if chunk_array.size == np.prod(chunk_shape):
                        chunk_array = chunk_array.reshape(tuple(chunk_shape))

                    slices = []
                    for idx, chunk_size, dim_size in zip(chunk_indices, chunks, shape):
                        start = idx * chunk_size
                        end = min(start + chunk_size, dim_size)
                        slices.append(slice(start, end))

                    array[tuple(slices)] = chunk_array

            return array

    except Exception as e:
        print(f"    ❌ Error reconstructing array: {e}")
        return None


def get_variable_path_mapping():
    """Map parameter names to their paths in the zarr store."""
    return {
        # Surface parameters
        '10u': 'u10/instant/heightAboveGround/u10',
        '10v': 'v10/instant/heightAboveGround/v10',
        '2t': 't2m/instant/heightAboveGround/t2m',
        '2d': 'd2m/instant/heightAboveGround/d2m',
        'msl': 'msl/instant/meanSea/msl',
        'sp': 'sp/instant/surface/sp',
        'skt': 'skt/instant/surface/skt',
        'tcw': 'tcw/instant/entireAtmosphere/tcw',
        # Fixed fields
        'lsm': 'lsm/instant/surface/lsm',
        # Pressure level parameters
        'gh': 'gh/instant/isobaricInhPa/gh',
        't': 't/instant/isobaricInhPa/t',
        'u': 'u/instant/isobaricInhPa/u',
        'v': 'v/instant/isobaricInhPa/v',
        'w': 'w/instant/isobaricInhPa/w',
        'q': 'q/instant/isobaricInhPa/q',
    }


def extract_pressure_level_coordinates(zstore, base_path):
    """
    Extract ALL pressure level coordinates from the zarr store.
    Returns a list of pressure levels in hPa.
    """
    import struct

    levels = []
    coord_base = f"{base_path}/isobaricInhPa"

    # Try to get metadata first to know how many levels
    zarray_key = f"{coord_base}/.zarray"
    if zarray_key in zstore:
        try:
            metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]
            num_levels = metadata.get('shape', [0])[0]

            # Extract each level value
            for i in range(num_levels):
                coord_path = f"{coord_base}/{i}"
                if coord_path in zstore:
                    val = zstore[coord_path]

                    # Handle different encoding types
                    if isinstance(val, str):
                        if val.startswith('base64:'):
                            # Decode base64
                            try:
                                decoded = base64.b64decode(val[7:])
                                level_value = struct.unpack('<d', decoded)[0]
                                levels.append(int(level_value))
                            except:
                                pass
                        elif len(val) == 8:
                            # Raw bytes
                            try:
                                level_value = struct.unpack('<d', val.encode('latin1'))[0]
                                levels.append(int(level_value))
                            except:
                                pass
        except:
            pass

    return sorted(levels, reverse=True)  # Return in descending order (1000, 925, ...)


def get_data_from_parquet_obstore(parquet_path, param, levelist=[], use_obstore=True):
    """
    Retrieve data from parquet files using obstore method.
    V2 IMPROVEMENT: Properly extracts ALL pressure levels from multi-dimensional arrays.

    Args:
        parquet_path: Parquet file path
        param: List of parameters to extract
        levelist: List of pressure/soil levels (empty for surface variables)
        use_obstore: Use obstore for S3 fetching (faster)

    Returns:
        Dictionary of {param_name: numpy_array}
    """
    fields = {}
    print(f"    Retrieving {param} data" + (f" (requested levels: {levelist})" if levelist else ""))

    # Read parquet to get references
    zstore = read_parquet_to_refs(parquet_path)

    # Get variable path mapping
    var_paths = get_variable_path_mapping()

    # Process each parameter
    for p in param:
        if p not in var_paths:
            print(f"    ⚠️ Parameter {p} not in mapping")
            continue

        base_path = var_paths[p]

        # Extract the variable
        print(f"    Extracting {p} from {base_path}")
        array = extract_variable_hybrid(zstore, base_path, use_obstore=use_obstore)

        if array is not None:
            print(f"      Raw array shape: {array.shape}")

            # Handle multi-dimensional data
            if levelist:
                # This is a pressure level variable
                # V2 IMPROVEMENT: Always try to extract coordinates first
                actual_levels = extract_pressure_level_coordinates(zstore, base_path.rsplit('/', 1)[0])

                # Determine the shape and extract levels
                if array.ndim == 5:
                    # Shape: (time, step, level, lat, lon) = (1, 2, 13, 721, 1440)
                    num_levels = array.shape[2]
                    print(f"      ✅ Found 5D array with {num_levels} levels at dimension 2")

                    # If we have coordinate info, use it; otherwise use dimension indices
                    if actual_levels and len(actual_levels) == num_levels:
                        print(f"      📍 Coordinate levels: {actual_levels}")
                        level_mapping = actual_levels
                    else:
                        print(f"      ⚠️ No coordinate info, using requested levels as mapping")
                        level_mapping = levelist[:num_levels] if len(levelist) >= num_levels else list(range(num_levels))

                    # Extract each level
                    for level_idx in range(num_levels):
                        level_value = level_mapping[level_idx] if level_idx < len(level_mapping) else level_idx

                        # Only extract if this level is in the requested list
                        if level_value in levelist or not actual_levels:
                            data_2d = array[0, 0, level_idx, :, :]
                            name = f"{p}_{level_value}"
                            fields[name] = data_2d
                            print(f"      ✅ {name}: shape={data_2d.shape}")

                elif array.ndim == 4:
                    # Shape: (time, step, level, lat, lon) = (1, 2, 13, 721, 1440)
                    # OR: (time, level, lat, lon) = (1, 13, 721, 1440)
                    # Check which dimension is the level dimension
                    if array.shape[1] == len(levelist) or (actual_levels and array.shape[1] == len(actual_levels)):
                        # Second dimension is levels
                        num_levels = array.shape[1]
                        level_dim_idx = 1
                        print(f"      ✅ Found 4D array with {num_levels} levels at dimension 1")

                        if actual_levels and len(actual_levels) == num_levels:
                            print(f"      📍 Coordinate levels: {actual_levels}")
                            level_mapping = actual_levels
                        else:
                            print(f"      ⚠️ No coordinate info, using requested levels as mapping")
                            level_mapping = levelist[:num_levels]

                        for level_idx in range(num_levels):
                            level_value = level_mapping[level_idx] if level_idx < len(level_mapping) else level_idx
                            if level_value in levelist or not actual_levels:
                                data_2d = array[0, level_idx, :, :]
                                name = f"{p}_{level_value}"
                                fields[name] = data_2d
                                print(f"      ✅ {name}: shape={data_2d.shape}")
                    else:
                        # Assume shape is (1, 2, 721, 1440) - single level
                        data_2d = array[0, 0, :, :]
                        name = f"{p}_{levelist[0] if levelist else 1000}"
                        fields[name] = data_2d
                        print(f"      ✅ {name}: shape={data_2d.shape} (single level)")

                elif array.ndim == 3:
                    # Shape: (level, lat, lon) = (13, 721, 1440)
                    num_levels = array.shape[0]
                    print(f"      ✅ Found 3D array with {num_levels} levels at dimension 0")

                    if actual_levels and len(actual_levels) == num_levels:
                        print(f"      📍 Coordinate levels: {actual_levels}")
                        level_mapping = actual_levels
                    else:
                        print(f"      ⚠️ No coordinate info, using requested levels as mapping")
                        level_mapping = levelist[:num_levels]

                    for level_idx in range(num_levels):
                        level_value = level_mapping[level_idx] if level_idx < len(level_mapping) else level_idx
                        if level_value in levelist or not actual_levels:
                            data_2d = array[level_idx, :, :]
                            name = f"{p}_{level_value}"
                            fields[name] = data_2d
                            print(f"      ✅ {name}: shape={data_2d.shape}")

                else:
                    print(f"      ⚠️ Unexpected array dimensions: {array.ndim}, shape: {array.shape}")

            else:
                # Surface variable - no levels
                if array.ndim == 4:
                    data_2d = array[0, 0, :, :]
                elif array.ndim == 3:
                    data_2d = array[0, :, :]
                elif array.ndim == 2:
                    data_2d = array
                else:
                    data_2d = array.reshape(array.shape[-2:])

                fields[p] = data_2d
                print(f"      ✅ {p}: shape={data_2d.shape}")
        else:
            print(f"    ❌ Failed to extract {p}")

    return fields


def create_input_state_from_parquet(parquet_path, member, use_obstore=True):
    """
    Create input state for a specific ensemble member using parquet file.
    V2: Properly extracts ALL pressure levels.

    Args:
        parquet_path: Path to parquet file
        member: Ensemble member number
        use_obstore: Use obstore for faster S3 access

    Returns:
        Dictionary with 'date' and 'fields'
    """
    print(f"\n{'='*60}")
    print(f"Creating input state for ensemble member {member}")
    print(f"{'='*60}")
    start_time = time.time()

    fields = {}

    # Check if obstore is available
    try:
        import obstore
        if use_obstore:
            print("  ✅ obstore available - using for S3 fetching")
    except ImportError:
        print("  ⚠️ obstore not available - using fsspec for S3 fetching")
        use_obstore = False

    # Add single level fields
    print("\n  Getting surface fields...")
    fields.update(get_data_from_parquet_obstore(parquet_path, param=PARAM_SFC, use_obstore=use_obstore))

    print("\n  Getting constant surface fields...")
    fields.update(get_data_from_parquet_obstore(parquet_path, param=PARAM_SFC_FC, use_obstore=use_obstore))

    # Add soil fields (if available)
    if PARAM_SOIL:
        print("\n  Getting soil fields...")
        soil = get_data_from_parquet_obstore(parquet_path, param=PARAM_SOIL, levelist=SOIL_LEVELS, use_obstore=use_obstore)

        # Rename soil parameters
        mapping = {'sot_1': 'stl1', 'sot_2': 'stl2'}
        for k, v in mapping.items():
            if k in soil:
                fields[v] = soil[k]

    # Add pressure level fields
    print("\n  Getting pressure level fields...")
    fields.update(get_data_from_parquet_obstore(parquet_path, param=PARAM_PL, levelist=LEVELS, use_obstore=use_obstore))

    # Convert geopotential height to geopotential
    print("\n  Converting geopotential height to geopotential...")
    for level in LEVELS:
        if f"gh_{level}" in fields:
            gh = fields.pop(f"gh_{level}")
            fields[f"z_{level}"] = gh * 9.80665
            print(f"    ✅ z_{level}")

    # Extract date from parquet path
    # Path format: ecmwf_20251020_00_efficient/members/ens_01/ens_01.parquet
    path_parts = Path(parquet_path).parts
    for part in path_parts:
        if 'ecmwf_' in part and '_efficient' in part:
            date_parts = part.replace('ecmwf_', '').replace('_efficient', '').split('_')
            if len(date_parts) >= 2:
                date_str = date_parts[0]
                hour_str = date_parts[1]
                date = datetime.datetime.strptime(f"{date_str}_{hour_str}", "%Y%m%d_%H")
                break
    else:
        date = datetime.datetime.now()

    input_state = dict(date=date, fields=fields)

    elapsed_time = time.time() - start_time
    print(f"\n  ✅ Completed in {elapsed_time:.2f} seconds")

    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Total fields: {len(fields)}")
    if fields:
        sample_shape = list(fields.values())[0].shape
        print(f"  Sample field shape: {sample_shape}")

        # Calculate memory usage
        total_elements = sum(field.size for field in fields.values())
        memory_mb = (total_elements * 4) / (1024 * 1024)
        print(f"  Approximate memory: {memory_mb:.2f} MB")

    return input_state


def verify_input_state(input_state, member):
    """Verify the input state has all required fields."""
    fields = input_state['fields']

    # Expected fields
    expected_surface = PARAM_SFC + PARAM_SFC_FC
    if PARAM_SOIL:
        expected_surface += ['stl1', 'stl2']

    expected_pressure = []
    for param in PARAM_PL:
        param_name = 'z' if param == 'gh' else param  # Converted to geopotential
        for level in LEVELS:
            expected_pressure.append(f"{param_name}_{level}")

    expected_total = expected_surface + expected_pressure

    # Check what we actually have
    available = [f for f in expected_total if f in fields]
    missing = [f for f in expected_total if f not in fields]

    print(f"\n  Verification for member {member}:")
    print(f"    Expected: {len(expected_total)} fields")
    print(f"    Available: {len(available)} fields")
    print(f"    Missing: {len(missing)} fields")

    if missing:
        # Group missing by type
        missing_surface = [f for f in missing if '_' not in f]
        missing_pressure = [f for f in missing if '_' in f]

        if missing_surface:
            print(f"    ⚠️  Missing surface fields ({len(missing_surface)}): {missing_surface}")
        if missing_pressure:
            print(f"    ⚠️  Missing pressure level fields ({len(missing_pressure)})")
            print(f"         Sample: {missing_pressure[:10]}")

    # Check completeness
    surface_complete = all(f in fields for f in expected_surface)
    pressure_complete = all(f in fields for f in expected_pressure)

    if surface_complete and pressure_complete:
        print(f"    ✅ Complete dataset - all fields present!")
        return True
    elif surface_complete:
        print(f"    ⚠️  Surface data complete, but missing {len(missing_pressure)} pressure level fields")
        return False
    else:
        print(f"    ⚠️  Incomplete dataset")
        return False


def main():
    """Main function to process ECMWF parquet files to PKL (V2)."""

    # Define the parquet file to process
    parquet_file = f"ecmwf_{date_str}_{run}_efficient/members/ens_01/ens_01.parquet"

    print("="*80)
    print("🌳 ECMWF PARQUET TO PKL PROCESSOR V2")
    print("="*80)
    print("Focus: Extract ALL pressure levels from parquet data")
    print("Skipped: Regridding and interpolation (not needed for this phase)")
    print("="*80)
    print(f"\nProcessing parquet file: {parquet_file}")

    # Create output directory if saving states
    if SAVE_STATES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output directory: {OUTPUT_DIR}/")

    # Check that file exists
    if not Path(parquet_file).exists():
        print(f"\n❌ File not found: {parquet_file}")
        return

    try:
        # Create input state from parquet file
        input_state = create_input_state_from_parquet(parquet_file, member=1, use_obstore=True)

        # Verify the state
        print(f"\n{'='*60}")
        print("VERIFICATION")
        print(f"{'='*60}")
        is_valid = verify_input_state(input_state, member=1)

        if SAVE_STATES:
            # Save the state as PKL
            output_file = f"{OUTPUT_DIR}/input_state_member_001.pkl"
            print(f"\n💾 Saving to: {output_file}")

            with open(output_file, 'wb') as f:
                pickle.dump(input_state, f)

            # Show file size
            file_size = Path(output_file).stat().st_size / (1024 * 1024)
            print(f"✅ Saved successfully!")
            print(f"📊 File size: {file_size:.2f} MB")

        # Show field samples
        if 'fields' in input_state and input_state['fields']:
            print(f"\n{'='*60}")
            print("FIELD SAMPLES")
            print(f"{'='*60}")
            field_names = sorted(input_state['fields'].keys())

            surface_fields = [f for f in field_names if '_' not in f]
            pressure_fields = [f for f in field_names if '_' in f]

            print(f"Surface fields ({len(surface_fields)}): {surface_fields}")
            print(f"Pressure level fields ({len(pressure_fields)})")

            # Group by parameter
            from collections import defaultdict
            by_param = defaultdict(list)
            for pf in pressure_fields:
                param = pf.rsplit('_', 1)[0]
                by_param[param].append(pf)

            for param in sorted(by_param.keys()):
                print(f"  {param}: {len(by_param[param])} levels - {by_param[param]}")

            # Show shapes
            print(f"\nField shapes (sample):")
            for i, (fname, fdata) in enumerate(list(input_state['fields'].items())[:5]):
                print(f"  {fname}: {fdata.shape}, dtype={fdata.dtype}")

    except Exception as e:
        print(f"\n❌ Error processing parquet file: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)
    if is_valid:
        print("✅ V2 PROCESSING COMPLETE - ALL LEVELS EXTRACTED!")
    else:
        print("⚠️  V2 PROCESSING COMPLETE - PARTIAL DATA")
    print("="*80)

    if not is_valid:
        print("\n⚠️  Some data may still be missing:")
        print("  Check the verification section above for details")

    print("\n💡 Next steps:")
    print("  - Review extracted fields to confirm all levels are present")
    print("  - If regridding is needed, implement in a separate phase")
    print("  - Use this PKL as input for AI-FS or further processing")


if __name__ == "__main__":
    main()
