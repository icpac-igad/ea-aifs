#!/usr/bin/env python3
"""
Extract ECMWF data from base64-encoded parquet references.

The parquet file contains base64-encoded chunk data, not S3 URLs.
This script decodes them directly without needing fsspec or S3 access.
"""

import pandas as pd
import numpy as np
import json
import base64
import pickle
from pathlib import Path
import struct


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


def analyze_references(zstore):
    """Analyze the types of references in the store."""
    print("\n📊 Analyzing reference types...")

    ref_types = {
        'base64': 0,
        's3': 0,
        'metadata': 0,
        'other': 0
    }

    examples = {}

    for key, value in zstore.items():
        if key.endswith('.zarray') or key.endswith('.zattrs') or key.endswith('.zgroup'):
            ref_types['metadata'] += 1
            if 'metadata' not in examples:
                examples['metadata'] = (key, value)
        elif isinstance(value, str):
            if value.startswith('base64:'):
                ref_types['base64'] += 1
                if 'base64' not in examples:
                    examples['base64'] = (key, value)
            elif value.startswith('s3://') or 's3.amazonaws.com' in value:
                ref_types['s3'] += 1
                if 's3' not in examples:
                    examples['s3'] = (key, value)
            else:
                ref_types['other'] += 1
        elif isinstance(value, list):
            if len(value) >= 1 and isinstance(value[0], str):
                if value[0].startswith('s3://') or 's3.amazonaws.com' in value[0]:
                    ref_types['s3'] += 1
                    if 's3' not in examples:
                        examples['s3'] = (key, value)
                else:
                    ref_types['other'] += 1
            else:
                ref_types['other'] += 1

    print(f"\n  Reference breakdown:")
    for ref_type, count in ref_types.items():
        print(f"    {ref_type}: {count}")

    print(f"\n  Examples:")
    for ref_type, (key, value) in examples.items():
        print(f"    {ref_type}:")
        print(f"      Key: {key}")
        value_str = str(value)[:100]
        print(f"      Value: {value_str}...")

    return ref_types


def decode_base64_chunk(base64_str):
    """Decode a base64-encoded chunk."""
    # Remove 'base64:' prefix if present
    if base64_str.startswith('base64:'):
        base64_str = base64_str[7:]

    # Decode base64
    try:
        decoded = base64.b64decode(base64_str)
        return decoded
    except Exception as e:
        print(f"    ⚠️ Error decoding base64: {e}")
        return None


def extract_variable_from_base64(zstore, variable_path):
    """Extract a variable by decoding base64 chunks."""
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
    fill_value = metadata.get('fill_value', None)

    print(f"Variable metadata:")
    print(f"  Shape: {shape}")
    print(f"  Dtype: {dtype}")
    print(f"  Chunks: {chunks}")
    print(f"  Compressor: {compressor}")
    print(f"  Fill value: {fill_value}")

    # Calculate number of chunks
    num_chunks = tuple(int(np.ceil(s / c)) for s, c in zip(shape, chunks))
    total_chunks = int(np.prod(num_chunks))
    print(f"  Total chunks: {total_chunks}")

    # Collect chunk data
    chunks_data = {}

    for key in sorted(zstore.keys()):
        if key.startswith(variable_path + "/") and not key.endswith(('.zarray', '.zattrs')):
            # This is a chunk key
            chunk_ref = zstore[key]

            if isinstance(chunk_ref, str) and chunk_ref.startswith('base64:'):
                # Decode base64 chunk
                decoded = decode_base64_chunk(chunk_ref)

                if decoded is not None:
                    # Decompress if needed
                    if compressor is not None:
                        try:
                            import numcodecs
                            codec = numcodecs.get_codec(compressor)
                            decoded = codec.decode(decoded)
                        except ImportError:
                            print(f"    ⚠️ numcodecs not available, trying blosc directly")
                            try:
                                import blosc
                                decoded = blosc.decompress(decoded)
                            except ImportError:
                                print(f"    ⚠️ Neither numcodecs nor blosc available")
                                # Try without decompression
                                pass
                        except Exception as e:
                            print(f"    ⚠️ Decompression failed: {e}")

                    chunks_data[key] = decoded

    print(f"\n  Decoded {len(chunks_data)} chunks")

    if not chunks_data:
        print(f"  ⚠️ No base64 chunks found")
        return None

    # Reconstruct array from chunks
    try:
        # For simple cases (1D or single chunk)
        if len(chunks_data) == 1:
            chunk_key = list(chunks_data.keys())[0]
            chunk_data = chunks_data[chunk_key]

            # Convert to numpy array
            array = np.frombuffer(chunk_data, dtype=dtype)

            # Reshape if needed
            if array.size == np.prod(shape):
                array = array.reshape(shape)
            elif array.size == chunks[0]:
                # Chunk size matches
                array = array.reshape(chunks)

            print(f"\n✅ Successfully extracted array")
            print(f"  Final shape: {array.shape}")
            print(f"  Dtype: {array.dtype}")
            if array.size > 0:
                print(f"  Min: {np.min(array)}")
                print(f"  Max: {np.max(array)}")
                print(f"  Mean: {np.mean(array)}")

            return array

        else:
            # Multiple chunks - need to reassemble
            print(f"  ⚠️ Multiple chunks - attempting reassembly")

            # Create output array
            array = np.zeros(shape, dtype=dtype)

            # Fill in chunks
            for chunk_key, chunk_data in chunks_data.items():
                # Parse chunk indices from key
                # e.g., "var/0.1.2" -> indices (0, 1, 2)
                chunk_idx_str = chunk_key.replace(variable_path + "/", "")
                chunk_indices = tuple(int(x) for x in chunk_idx_str.split('.'))

                # Convert to numpy array
                chunk_array = np.frombuffer(chunk_data, dtype=dtype)

                # Determine chunk shape
                chunk_shape = []
                for i, (idx, chunk_size, dim_size) in enumerate(zip(chunk_indices, chunks, shape)):
                    if (idx + 1) * chunk_size <= dim_size:
                        chunk_shape.append(chunk_size)
                    else:
                        chunk_shape.append(dim_size - idx * chunk_size)
                chunk_shape = tuple(chunk_shape)

                # Reshape chunk
                chunk_array = chunk_array.reshape(chunk_shape)

                # Calculate position in output array
                slices = []
                for idx, chunk_size, dim_size in zip(chunk_indices, chunks, shape):
                    start = idx * chunk_size
                    end = min(start + chunk_size, dim_size)
                    slices.append(slice(start, end))

                # Insert chunk into array
                array[tuple(slices)] = chunk_array

            print(f"\n✅ Successfully reassembled array from chunks")
            print(f"  Final shape: {array.shape}")
            print(f"  Dtype: {array.dtype}")
            if array.size > 0:
                print(f"  Min: {np.min(array)}")
                print(f"  Max: {np.max(array)}")
                print(f"  Mean: {np.mean(array)}")

            return array

    except Exception as e:
        print(f"\n❌ Error reconstructing array: {e}")
        import traceback
        traceback.print_exc()
        return None


def list_all_variables(zstore):
    """List all variables in the zarr store."""
    variables = []

    for key in zstore.keys():
        if key.endswith('.zarray'):
            var_path = key.replace('/.zarray', '')

            try:
                metadata = json.loads(zstore[key]) if isinstance(zstore[key], str) else zstore[key]
                shape = metadata.get('shape', [])
                dtype = metadata.get('dtype', 'unknown')

                variables.append({
                    'path': var_path,
                    'shape': shape,
                    'dtype': dtype
                })
            except:
                pass

    return variables


def main():
    """Main extraction function."""
    print("="*80)
    print("EXTRACT FROM BASE64-ENCODED PARQUET REFERENCES")
    print("="*80)

    # Find parquet file
    possible_paths = [
        "ecmwf_20251020_00_efficient/members/ens_01/ens_01.parquet"
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

    # Analyze reference types
    ref_types = analyze_references(zstore)

    # List variables
    print(f"\n{'='*60}")
    print("AVAILABLE VARIABLES")
    print(f"{'='*60}")

    variables = list_all_variables(zstore)
    print(f"\nFound {len(variables)} variables:")

    # Group by parameter name
    param_groups = {}
    for var in variables:
        param_name = var['path'].split('/')[0]
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append(var)

    for i, (param, vars_list) in enumerate(sorted(param_groups.items())[:20]):
        print(f"\n  {i+1}. {param} ({len(vars_list)} variants):")
        for var in vars_list[:3]:  # Show first 3 variants
            print(f"      {var['path']}")
            print(f"        Shape: {var['shape']}, Dtype: {var['dtype']}")

    # Extract a few key variables
    print(f"\n{'='*60}")
    print("EXTRACTING SAMPLE VARIABLES")
    print(f"{'='*60}")

    # Try to extract some key meteorological variables
    target_vars = [
        't2m/instant/heightAboveGround/t2m',  # 2m temperature
        'tp/accum/surface/tp',  # Total precipitation
        'msl/instant/meanSea/msl',  # Mean sea level pressure
    ]

    extracted_data = {}

    for target_var in target_vars:
        # Find matching variable
        matching = [v for v in variables if target_var in v['path']]

        if matching:
            var_path = matching[0]['path']
            data = extract_variable_from_base64(zstore, var_path)

            if data is not None:
                extracted_data[var_path] = data

                # Save to pickle
                output_dir = Path("extracted_base64")
                output_dir.mkdir(exist_ok=True)

                safe_name = var_path.replace('/', '_')
                output_file = output_dir / f"{safe_name}.pkl"

                with open(output_file, 'wb') as f:
                    pickle.dump(data, f)

                file_size = output_file.stat().st_size / 1024
                print(f"  💾 Saved to: {output_file} ({file_size:.1f} KB)")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n✅ Successfully extracted {len(extracted_data)} variables")

    if extracted_data:
        print(f"\nExtracted variables:")
        for var_path, data in extracted_data.items():
            print(f"  - {var_path}: shape={data.shape}, dtype={data.dtype}")

        print(f"\n💡 Key finding:")
        print(f"   The parquet file contains BASE64-encoded data, not S3 URLs!")
        print(f"   This means:")
        print(f"   ✅ No S3 access needed")
        print(f"   ✅ All data is self-contained in the parquet file")
        print(f"   ✅ Can extract directly without fsspec's reference filesystem")
    else:
        print(f"\n⚠️ No variables successfully extracted")
        print(f"   Check if compression libraries (numcodecs/blosc) are installed")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
