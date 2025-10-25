#!/usr/bin/env python3
"""
Extract ECMWF data from hybrid parquet references (base64 + S3).

The parquet file contains:
- base64-encoded chunks for small metadata arrays
- S3 byte-range references for large data arrays

This script handles both types and shows where obstore could help.
"""

import pandas as pd
import numpy as np
import json
import base64
import pickle
from pathlib import Path
import io
import tempfile
import os


def read_parquet_to_refs(parquet_path):
    """Read parquet file and extract zarr references."""
    print(f"Reading parquet file: {parquet_path}")
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
    """
    Decode a chunk reference. Returns (type, data).

    Types:
    - 'base64': data is decoded bytes
    - 's3': data is (url, offset, length)
    - 'unknown': data is raw value
    """
    if isinstance(chunk_ref, str):
        if chunk_ref.startswith('base64:'):
            # Base64-encoded data
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


def fetch_s3_byte_range_fsspec(url, offset, length):
    """Fetch a byte range from S3 using fsspec."""
    try:
        import fsspec

        # Parse S3 URL
        if url.startswith('s3://'):
            s3_path = url[5:]  # Remove 's3://'
        else:
            s3_path = url

        # Open S3 filesystem
        fs = fsspec.filesystem('s3', anon=True)

        # Read the byte range
        with fs.open(s3_path, 'rb') as f:
            f.seek(offset)
            data = f.read(length)

        return data

    except Exception as e:
        print(f"    ❌ Error fetching from S3 with fsspec: {e}")
        return None


def fetch_s3_byte_range_obstore(url, offset, length):
    """
    Fetch a byte range from S3 using obstore (if available).
    Based on obstore API from working_virtualizarr_processor.py
    """
    try:
        import obstore as obs
        from obstore.store import from_url

        print(f"    📦 Using obstore for S3 fetch (faster)")

        # Parse bucket and key from URL
        if url.startswith('s3://'):
            url_parts = url[5:].split('/', 1)
            bucket = url_parts[0]
            key = url_parts[1] if len(url_parts) > 1 else ''
        else:
            raise ValueError(f"Invalid S3 URL: {url}")

        # Determine region based on bucket name
        # ECMWF buckets are in EU regions
        bucket_regions = {
            'ecmwf-forecasts': 'eu-central-1',
            'noaa-cdr-precip-cmorph-pds': 'us-east-1',
            'noaa-cdr-precip-persiann-pds': 'us-east-1',
        }
        region = bucket_regions.get(bucket, 'eu-central-1')  # Default to EU for ECMWF

        print(f"      Using region: {region} for bucket: {bucket}")

        # Create S3 store using from_url (following working pattern)
        bucket_url = f"s3://{bucket}"
        store = from_url(bucket_url, region=region, skip_signature=True)

        # Fetch byte range using obstore.get_range
        # API: get_range(store, path, start=offset, end=offset+length)
        result = obs.get_range(store, key, start=offset, end=offset + length)

        # Convert to bytes
        data = bytes(result)

        print(f"      ✅ Fetched {len(data):,} bytes via obstore")
        return data

    except ImportError:
        print(f"    ⚠️ obstore not available, falling back to fsspec")
        return fetch_s3_byte_range_fsspec(url, offset, length)
    except Exception as e:
        print(f"    ❌ Error fetching from S3 with obstore: {e}")
        print(f"    Falling back to fsspec...")
        return fetch_s3_byte_range_fsspec(url, offset, length)


def extract_variable_hybrid(zstore, variable_path, use_obstore=False):
    """Extract a variable handling both base64 and S3 references."""
    print(f"\n{'='*60}")
    print(f"Extracting variable: {variable_path}")
    print(f"{'='*60}")

    # Get metadata
    zarray_key = f"{variable_path}/.zarray"
    if zarray_key not in zstore:
        print(f"❌ No metadata found for {variable_path}")
        return None

    metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]

    shape = tuple(metadata['shape'])
    dtype = np.dtype(metadata['dtype'])
    chunks = tuple(metadata['chunks'])
    compressor = metadata.get('compressor', None)

    print(f"Variable metadata:")
    print(f"  Shape: {shape}")
    print(f"  Dtype: {dtype}")
    print(f"  Chunks: {chunks}")
    print(f"  Compressor: {compressor}")

    # Calculate number of chunks
    num_chunks = tuple(int(np.ceil(s / c)) for s, c in zip(shape, chunks))
    total_chunks = int(np.prod(num_chunks))
    print(f"  Total chunks: {total_chunks}")

    # Collect chunks
    chunks_data = {}
    chunk_types = {'base64': 0, 's3': 0, 'unknown': 0}

    for key in sorted(zstore.keys()):
        if key.startswith(variable_path + "/") and not key.endswith(('.zarray', '.zattrs', '.zgroup')):
            chunk_ref = zstore[key]
            ref_type, ref_data = decode_chunk_reference(chunk_ref)
            chunk_types[ref_type] += 1

            if ref_type == 'base64':
                # Decompress if needed
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
                print(f"    Fetching chunk from S3: {Path(url).name}")
                print(f"      Offset: {offset:,}, Length: {length:,}")

                # Fetch from S3
                if use_obstore:
                    data = fetch_s3_byte_range_obstore(url, offset, length)
                else:
                    data = fetch_s3_byte_range_fsspec(url, offset, length)

                if data is not None:
                    # The data is in GRIB2 format, need to decode it
                    print(f"      Fetched {len(data):,} bytes")

                    # Check if it's GRIB2 data
                    if data[:4] == b'GRIB':
                        print(f"      ✅ GRIB2 data detected")
                        # Decode GRIB2 message
                        try:
                            import cfgrib
                            import xarray as xr

                            # Write to temporary file for cfgrib
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
                                tmp.write(data)
                                tmp_path = tmp.name

                            # Open with cfgrib
                            ds = xr.open_dataset(tmp_path, engine='cfgrib')

                            # Extract the data array
                            var_names = list(ds.data_vars)
                            if var_names:
                                var_data = ds[var_names[0]].values
                                print(f"      Decoded shape: {var_data.shape}, dtype: {var_data.dtype}")

                                # Store the actual numpy array, not bytes
                                # This preserves the dtype (usually float32 from GRIB2)
                                chunks_data[key] = var_data
                            else:
                                print(f"      ⚠️ No variables found in GRIB2")

                            # Clean up
                            os.unlink(tmp_path)
                            ds.close()

                        except ImportError:
                            print(f"      ⚠️ cfgrib not available, trying eccodes")
                            try:
                                import eccodes

                                # Decode using eccodes
                                # Create a message from the data
                                gid = eccodes.codes_new_from_message(data)

                                # Get the values
                                values = eccodes.codes_get_array(gid, 'values')

                                # Release the message
                                eccodes.codes_release(gid)

                                print(f"      Decoded {len(values)} values, dtype: {values.dtype}")
                                # Store as numpy array
                                chunks_data[key] = values

                            except ImportError:
                                print(f"      ❌ Neither cfgrib nor eccodes available")
                                print(f"      Cannot decode GRIB2 data")
                            except Exception as e:
                                print(f"      ❌ Error decoding with eccodes: {e}")

                        except Exception as e:
                            print(f"      ❌ Error decoding GRIB2: {e}")

                    else:
                        # Not GRIB2, maybe compressed zarr chunk
                        print(f"      Data format: {data[:4]}")

                        # Try decompression if needed
                        if compressor is not None:
                            try:
                                import numcodecs
                                codec = numcodecs.get_codec(compressor)
                                data = codec.decode(data)
                                chunks_data[key] = data
                            except Exception as e:
                                print(f"      ⚠️ Decompression failed: {e}")
                        else:
                            chunks_data[key] = data
                else:
                    print(f"      ❌ Failed to fetch chunk")

    print(f"\n  Chunk breakdown:")
    for ctype, count in chunk_types.items():
        print(f"    {ctype}: {count}")
    print(f"  Successfully retrieved: {len(chunks_data)} chunks")

    if not chunks_data:
        print(f"  ⚠️ No chunks retrieved")
        return None

    # Reconstruct array
    try:
        if len(chunks_data) == 1:
            # Single chunk
            chunk_data = list(chunks_data.values())[0]

            # Check if it's already a numpy array (from GRIB2)
            if isinstance(chunk_data, np.ndarray):
                array = chunk_data
            else:
                array = np.frombuffer(chunk_data, dtype=dtype)

            if array.size == np.prod(shape):
                array = array.reshape(shape)

            print(f"\n✅ Successfully extracted array")
            print(f"  Shape: {array.shape}")
            print(f"  Dtype: {array.dtype}")
            if array.size > 0:
                print(f"  Range: [{np.min(array):.6f}, {np.max(array):.6f}]")
                print(f"  Mean: {np.mean(array):.6f}")

            return array

        else:
            # Multiple chunks - reassemble
            print(f"  Reassembling {len(chunks_data)} chunks...")

            # Determine actual dtype from first chunk
            first_chunk = list(chunks_data.values())[0]
            if isinstance(first_chunk, np.ndarray):
                actual_dtype = first_chunk.dtype
                print(f"    Using dtype from GRIB2 data: {actual_dtype}")
            else:
                actual_dtype = dtype

            array = np.zeros(shape, dtype=actual_dtype)

            for chunk_key, chunk_data in chunks_data.items():
                # Parse chunk indices
                chunk_idx_str = chunk_key.replace(variable_path + "/", "")
                chunk_indices = tuple(int(x) for x in chunk_idx_str.split('.'))

                # Convert to numpy if needed
                if isinstance(chunk_data, np.ndarray):
                    chunk_array = chunk_data
                else:
                    chunk_array = np.frombuffer(chunk_data, dtype=actual_dtype)

                print(f"    Chunk {chunk_indices}: shape={chunk_array.shape}, dtype={chunk_array.dtype}")

                # GRIB2 data comes as 2D (721, 1440), but metadata expects 4D (1, 1, 721, 1440)
                # The chunk indices tell us which (time, step) position to fill
                if chunk_array.ndim == 2 and len(shape) == 4:
                    # Insert the 2D GRIB2 data at the right (time, step) position
                    time_idx = chunk_indices[0] if len(chunk_indices) > 0 else 0
                    step_idx = chunk_indices[1] if len(chunk_indices) > 1 else 0
                    array[time_idx, step_idx, :, :] = chunk_array
                    print(f"      Inserted at position [{time_idx}, {step_idx}, :, :]")
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

                    # Insert into array
                    slices = []
                    for idx, chunk_size, dim_size in zip(chunk_indices, chunks, shape):
                        start = idx * chunk_size
                        end = min(start + chunk_size, dim_size)
                        slices.append(slice(start, end))

                    array[tuple(slices)] = chunk_array

            print(f"\n✅ Successfully reassembled array")
            print(f"  Shape: {array.shape}")
            print(f"  Dtype: {array.dtype}")
            if array.size > 0:
                print(f"  Range: [{np.min(array):.6f}, {np.max(array):.6f}]")
                print(f"  Mean: {np.mean(array):.6f}")

            return array

    except Exception as e:
        print(f"\n❌ Error reconstructing array: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main extraction function."""
    print("="*80)
    print("EXTRACT FROM HYBRID PARQUET REFERENCES (BASE64 + S3)")
    print("="*80)

    # Find parquet file
    possible_paths = [
        "ecmwf_20251020_00_efficient/members/ens_01/ens_01.parquet",
    ]

    parquet_path = None
    for path in possible_paths:
        if Path(path).exists():
            parquet_path = path
            break

    if not parquet_path:
        print("❌ Parquet file not found")
        return

    print(f"\nFile: {parquet_path}\n")

    # Read references
    zstore = read_parquet_to_refs(parquet_path)

    # Test with a sample variable
    print(f"\n{'='*60}")
    print("TESTING S3 EXTRACTION")
    print(f"{'='*60}")

    # Try to extract t2m (should have S3 references)
    test_var = 't2m/instant/heightAboveGround/t2m'

    print(f"\nAttempting to extract: {test_var}")
    print(f"This should demonstrate S3 byte-range fetching")

    # Check if obstore is available
    try:
        import obstore
        use_obstore = True
        print(f"\n✅ obstore is available - will use for S3 fetches")
    except ImportError:
        use_obstore = False
        print(f"\n⚠️ obstore not available - will use fsspec for S3 fetches")

    data = extract_variable_hybrid(zstore, test_var, use_obstore=use_obstore)

    if data is not None:
        # Save to pickle
        output_dir = Path("extracted_hybrid")
        output_dir.mkdir(exist_ok=True)

        safe_name = test_var.replace('/', '_')
        output_file = output_dir / f"{safe_name}.pkl"

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"\n💾 Saved to: {output_file} ({file_size:.2f} MB)")

        # Summary
        print(f"\n{'='*80}")
        print("SUCCESS!")
        print(f"{'='*80}")
        print(f"\n✅ Successfully extracted {test_var}")
        print(f"   Shape: {data.shape}")
        print(f"   Size: {file_size:.2f} MB")

        print(f"\n💡 This demonstrates:")
        print(f"   1. Parsing kerchunk parquet references")
        print(f"   2. Fetching S3 byte ranges")
        print(f"   3. Reconstructing zarr arrays")
        print(f"   4. {'Using obstore (Rust) for faster I/O' if use_obstore else 'Using fsspec (Python) for I/O'}")

        print(f"\n🚀 Next steps:")
        print(f"   - Extract all required ECMWF parameters")
        print(f"   - Apply preprocessing (interpolation, shifting)")
        print(f"   - Create PKL file in AI-FS format")

    else:
        print(f"\n❌ Failed to extract variable")
        print(f"\nPossible issues:")
        print(f"  - S3 access permissions")
        print(f"  - Network connectivity")
        print(f"  - Chunk reference format")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
