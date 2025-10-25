import fsspec
import pandas as pd
import numpy as np
import copy
import json
import zarr
from collections import defaultdict
from typing import Dict, List, Iterable, Set, Optional
import xarray as xr
from kerchunk.grib2 import scan_grib, grib_tree
from kerchunk.combine import MultiZarrToZarr
from kerchunk._grib_idx import strip_datavar_chunks
import pickle
import os
import datetime
import time
"""
fsspec: 2025.5.1
pandas: 2.3.0
numpy: 2.3.0
zarr: 2.18.7
xarray: 2025.6.1
kerchunk: 0.2.7
"""


def log_checkpoint(message: str, start_time: float = None):
    """Log a checkpoint with timestamp and elapsed time."""
    current_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if start_time is not None:
        elapsed = current_time - start_time
        print(f"[{timestamp}] {message} (Elapsed: {elapsed:.2f}s)")
    else:
        print(f"[{timestamp}] {message}")

    return current_time


def s3_parse_ecmwf_grib_idx(
    fs: fsspec.AbstractFileSystem,
    basename: str,
    suffix: str = "index",
    tstamp: Optional[pd.Timestamp] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """
    Standalone method used to extract metadata from a grib2 index file

    :param fs: the file system to read from
    :param basename: the base name is the full path to the grib file
    :param suffix: the suffix is the ending for the index file
    :param tstamp: the timestamp to record for this index process
    :return: the data frame containing the results
    """
    fname = f"{basename.rsplit('.', 1)[0]}.{suffix}"

    fs.invalidate_cache(fname)
    fs.invalidate_cache(basename)

    baseinfo = fs.info(basename)

    with fs.open(fname, "r") as f:
        splits = []
        for idx, line in enumerate(f):
            try:
                # Removing the trailing characters if there's any at the end of the line
                clean_line = line.strip().rstrip(',')
                # Convert the JSON-like string to a dictionary
                data = json.loads(clean_line)
                # Extracting required fields using .get() method to handle missing keys
                lidx = idx
                offset = data.get("_offset", 0)  # Default to 0 if missing
                length = data.get("_length", 0)
                date = data.get(
                    "date",
                    "Unknown Date")  # Default to 'Unknown Date' if missing
                ens_number = data.get("number", -1)  # Default to -1 if missing
                # Append to the list as integers or the original data type
                splits.append([
                    int(lidx),
                    int(offset),
                    int(length), date, data,
                    int(ens_number)
                ])
            except json.JSONDecodeError as e:
                # Handle cases where JSON conversion fails
                raise ValueError(
                    f"Could not parse JSON from line: {line}") from e

    result = pd.DataFrame(
        splits,
        columns=["idx", "offset", "length", "date", "attr", "ens_number"])

    # Subtract the next offset to get the length using the filesize for the last value

    result.loc[:, "idx_uri"] = fname
    result.loc[:, "grib_uri"] = basename

    if tstamp is None:
        tstamp = pd.Timestamp.now()
    #result.loc[:, "indexed_at"] = tstamp
    result['indexed_at'] = result.apply(lambda x: tstamp, axis=1)

    # Check for S3 or GCS filesystem instances to handle metadata
    if "s3" in fs.protocol:
        # Use ETag as the S3 equivalent to crc32c
        result.loc[:, "grib_etag"] = baseinfo.get("ETag")
        result.loc[:, "grib_updated_at"] = pd.to_datetime(
            baseinfo.get("LastModified")).tz_localize(None)

        idxinfo = fs.info(fname)
        result.loc[:, "idx_etag"] = idxinfo.get("ETag")
        result.loc[:, "idx_updated_at"] = pd.to_datetime(
            idxinfo.get("LastModified")).tz_localize(None)
    else:
        # TODO: Fix metadata for other filesystems
        result.loc[:, "grib_crc32"] = None
        result.loc[:, "grib_updated_at"] = None
        result.loc[:, "idx_crc32"] = None
        result.loc[:, "idx_updated_at"] = None

    if validate and not result["attrs"].is_unique:
        raise ValueError(
            f"Attribute mapping for grib file {basename} is not unique)")
    print(f'Completed index files and found {len(result.index)} entries in it')
    return result.set_index("idx")


def ecmwf_idx_unique_dict(edf):
    """
    Extract unique parameter combinations from ECMWF index dataframe.

    FIXED: Now includes ALL pressure levels (50, 100, 150, 200, 250, 300, 400, 500,
    600, 700, 850, 925, 1000) and ALL soil levels (1, 2, 4), not just level 50.
    """
    # Fill empty rows or missing values in 'levelist' with 'null'
    edf['levelist'] = edf['levelist'].fillna('null')

    # FIXED: Include ALL pressure levels and soil levels, not just 50!
    # Define the levels we want to extract
    pressure_levels = ['50', '100', '150', '200', '250', '300', '400', '500',
                      '600', '700', '850', '925', '1000']
    soil_levels = ['1', '2', '4']

    # Filter for:
    # 1. ALL pressure levels (pl)
    # 2. ALL soil levels (sol)
    # 3. Surface parameters (sfc)
    combined_params = edf[
        ((edf['levtype'] == 'pl') & (edf['levelist'].isin(pressure_levels))) |
        ((edf['levtype'] == 'sol') & (edf['levelist'].isin(soil_levels))) |
        (edf['levtype'] == 'sfc')
    ].groupby(['param', 'levtype', 'levelist']).agg({
        'ens_number': lambda x: -1 if (-1 in x.values) else x.iloc[0]
    }).reset_index()

    combined_dict = {}
    for _, row in combined_params.iterrows():
        # Include level in the key for pressure/soil levels to keep them separate
        if row['levtype'] in ['pl', 'sol']:
            key = f"{row['param']}_{row['levtype']}_{row['levelist']}"
        else:
            key = f"{row['param']}_{row['levtype']}"

        combined_dict[key] = {
            'param': row['param'],
            'levtype': row['levtype'],
            'ens_number': row['ens_number'],
            'levelist': row['levelist']
        }
    return combined_dict


def ecmwf_duplicate_dict_ens_mem(var_dict):
    # Generate sequence for ensemble members 1-50, with control (-1) at the start
    ens_numbers = np.arange(1, 51)
    ens_numbers = np.insert(ens_numbers, 0, -1)
    updated_data_dict = var_dict.copy()
    for ens_number in ens_numbers:
        for key, subdict in var_dict.items():
            updated_subdict = subdict.copy()
            updated_subdict['ens_number'] = int(ens_number)
            new_key = f"{key}_ens{ens_number}"
            updated_data_dict[new_key] = updated_subdict
    return updated_data_dict


def ecmwf_get_matching_indices(filter_dict, df):
    # Get the indices of rows in a DataFrame that match the criteria in a dictionary.
    filtered_dfs = []
    for key, conditions in filter_dict.items():
        mask = True
        for col, value in conditions.items():
            if value == 'null':
                mask = mask & (df[col] == 'null')
            else:
                mask = mask & (df[col] == value)
        filtered_df = df[mask]
        if not filtered_df.empty:
            filtered_dfs.append(filtered_df)
    final_df = pd.concat(filtered_dfs).sort_values(['param', 'levtype'])
    return final_df.index.tolist()


def ecmwf_idx_df_create_with_keys(ecmwf_s3url):
    fs = fsspec.filesystem("s3", anon=True)
    suffix = 'index'
    idx_file_index = s3_parse_ecmwf_grib_idx(fs=fs,
                                             basename=ecmwf_s3url,
                                             suffix=suffix)
    edf = pd.concat([
        idx_file_index.drop('attr', axis=1), idx_file_index['attr'].apply(
            pd.Series)
    ],
                    axis=1)
    combined_dict = ecmwf_idx_unique_dict(edf)
    all_em = ecmwf_duplicate_dict_ens_mem(combined_dict)
    idx_mapping = {}
    for ens_key, conditions in all_em.items():
        mask = True
        for col, value in conditions.items():
            if value == 'null':
                mask = mask & (edf[col] == 'null')
            else:
                mask = mask & (edf[col] == value)
        matching_indices = edf[mask].index.tolist()
        for idx in matching_indices:
            idx_mapping[idx] = ens_key
    return idx_mapping, combined_dict


def ecmwf_filter_scan_grib(ecmwf_s3url):
    """
    Scan an ECMWF GRIB file, add ensemble information to the Zarr references,
    and return a list of modified groups along with an index mapping.
    """
    # Use anonymous access for S3
    storage_options = {"anon": True}
    esc_groups = scan_grib(ecmwf_s3url, storage_options=storage_options)
    print(
        f"Completed scan_grib for {ecmwf_s3url}, found {len(esc_groups)} messages"
    )
    idx_mapping, _ = ecmwf_idx_df_create_with_keys(ecmwf_s3url)
    print(f"Found {len(idx_mapping)} matching indices")
    modified_groups = []
    for i, group in enumerate(esc_groups):
        if i in idx_mapping:
            ens_key = idx_mapping[i]
            ens_number = int(
                ens_key.split('ens')[-1]) if 'ens' in ens_key else -1
            mod_group = copy.deepcopy(group)
            refs = mod_group['refs']
            data_vars = []
            for key in refs:
                if key.endswith('/.zattrs'):
                    var_name = key.split('/')[0]
                    if not var_name.startswith('.'):
                        try:
                            attrs = json.loads(refs[key])
                            if '_ARRAY_DIMENSIONS' in attrs and len(
                                    attrs['_ARRAY_DIMENSIONS']) > 0:
                                if var_name not in [
                                        'latitude', 'longitude', 'number',
                                        'time', 'step', 'valid_time'
                                ]:
                                    data_vars.append(var_name)
                        except json.JSONDecodeError:
                            print(f"Error decoding {key}")
            if '.zattrs' in refs:
                try:
                    root_attrs = json.loads(refs['.zattrs'])
                    root_attrs['ensemble_member'] = ens_number
                    root_attrs['ensemble_key'] = ens_key
                    if 'coordinates' in root_attrs:
                        coords = root_attrs['coordinates'].split()
                        if 'number' not in coords:
                            coords.append('number')
                            root_attrs['coordinates'] = ' '.join(coords)
                    refs['.zattrs'] = json.dumps(root_attrs)
                except json.JSONDecodeError:
                    print(f"Error updating root attributes for group {i}")
            for var_name in data_vars:
                attr_key = f"{var_name}/.zattrs"
                if attr_key in refs:
                    try:
                        var_attrs = json.loads(refs[attr_key])
                        var_attrs['ensemble_member'] = ens_number
                        var_attrs['ensemble_key'] = ens_key
                        refs[attr_key] = json.dumps(var_attrs)
                    except json.JSONDecodeError:
                        print(f"Error updating attributes for {var_name}")
            has_number = any(
                key == 'number/.zattrs' or key.endswith('/number/.zattrs')
                for key in refs)
            if not has_number:
                print(
                    f"Adding number coordinate for group {i}, ensemble {ens_number}"
                )
                refs['number/.zarray'] = json.dumps({
                    "chunks": [],
                    "compressor": None,
                    "dtype": "<i8",
                    "fill_value": None,
                    "filters": None,
                    "order": "C",
                    "shape": [],
                    "zarr_format": 2
                })
                refs['number/.zattrs'] = json.dumps({
                    "_ARRAY_DIMENSIONS": [],
                    "long_name": "ensemble member numerical id",
                    "standard_name": "realization",
                    "units": "1"
                })
                ens_num_array = np.array(ens_number, dtype=np.int64)
                refs['number/0'] = ens_num_array.tobytes().decode('latin1')
            modified_groups.append(mod_group)
    return modified_groups, idx_mapping


def organize_ensemble_tree(original_tree):
    """
    Reorganize the original Zarr tree by adding ensemble dimensions to the attributes.
    """
    ensemble_tree = copy.deepcopy(original_tree)
    attrs_keys = [k for k in ensemble_tree['refs'] if k.endswith('/.zattrs')]
    for key in attrs_keys:
        try:
            attrs = json.loads(ensemble_tree['refs'][key])
            if '_ARRAY_DIMENSIONS' in attrs:
                if 'number' not in attrs['_ARRAY_DIMENSIONS']:
                    dimensions = attrs['_ARRAY_DIMENSIONS']
                    if 'time' in dimensions and 'step' in dimensions:
                        step_index = dimensions.index('step')
                        dimensions.insert(step_index + 1, 'number')
                    else:
                        dimensions.insert(0, 'number')
                    attrs['_ARRAY_DIMENSIONS'] = dimensions
                    ensemble_tree['refs'][key] = json.dumps(attrs)
            if key == '.zattrs':
                if 'coordinates' in attrs:
                    coords = attrs['coordinates'].split()
                    if 'number' not in coords:
                        coords.append('number')
                        attrs['coordinates'] = ' '.join(coords)
                        ensemble_tree['refs'][key] = json.dumps(attrs)
        except json.JSONDecodeError:
            print(f"Error parsing attributes for {key}")
    if 'number/.zattrs' not in ensemble_tree['refs']:
        ensemble_tree['refs']['number/.zattrs'] = json.dumps({
            "_ARRAY_DIMENSIONS": [],
            "long_name":
            "ensemble member numerical id",
            "standard_name":
            "realization",
            "units":
            "1"
        })
        ensemble_tree['refs']['number/.zarray'] = json.dumps({
            "chunks": [51],
            "compressor": None,
            "dtype": "<i8",
            "fill_value": None,
            "filters": None,
            "order": "C",
            "shape": [51],
            "zarr_format": 2
        })
        numbers = np.arange(-1, 50, dtype=np.int64)
        ensemble_tree['refs']['number/0'] = numbers.tobytes().decode('latin1')
    array_keys = [k for k in ensemble_tree['refs'] if k.endswith('/.zarray')]
    for key in array_keys:
        try:
            var_name = key.split('/')[0]
            if var_name in [
                    'latitude', 'longitude', 'time', 'step', 'valid_time',
                    'number'
            ]:
                continue
            attr_key = key.replace('/.zarray', '/.zattrs')
            if attr_key in ensemble_tree['refs']:
                attrs = json.loads(ensemble_tree['refs'][attr_key])
                if '_ARRAY_DIMENSIONS' in attrs and 'number' in attrs[
                        '_ARRAY_DIMENSIONS']:
                    array_def = json.loads(ensemble_tree['refs'][key])
                    number_index = attrs['_ARRAY_DIMENSIONS'].index('number')
                    shape = array_def['shape']
                    if number_index < len(shape):
                        shape.insert(number_index, 51)
                    elif number_index == len(shape):
                        shape.append(51)
                    if 'chunks' in array_def and array_def['chunks']:
                        chunks = array_def['chunks']
                        if number_index < len(chunks):
                            chunks.insert(number_index, 51)
                        elif number_index == len(chunks):
                            chunks.append(51)
                    ensemble_tree['refs'][key] = json.dumps(array_def)
        except json.JSONDecodeError:
            print(f"Error parsing array definition for {key}")
    return ensemble_tree


def fixed_ensemble_grib_tree(message_groups: Iterable[Dict],
                             remote_options=None,
                             debug_output=False) -> Dict:
    """
    Build a hierarchical data model from a set of scanned grib messages with proper ensemble support
    and correct zarr path structure.
    
    This function handles ensemble dimensions correctly while maintaining the proper zarr structure
    needed by xr.datatree.
    
    Parameters
    ----------
    message_groups: iterable[dict]
        a collection of zarr store like dictionaries as produced by scan_grib
    remote_options: dict
        remote options to pass to MultiZarrToZarr
    debug_output: bool
        If True, prints detailed debugging information

    Returns
    -------
    dict: A zarr store like dictionary with proper ensemble support
    """
    # Set default remote options for anonymous S3 access
    if remote_options is None:
        remote_options = {"anon": True}
    # Hard code the filters in the correct order for the group hierarchy
    filters = ["stepType", "typeOfLevel"]

    # Use a regular dictionary for storage
    zarr_store = {'.zgroup': json.dumps({'zarr_format': 2})}
    zroot = zarr.group()

    # Track information by path
    aggregations = defaultdict(list)
    ensemble_dimensions = defaultdict(set)
    level_dimensions = defaultdict(set)
    path_counts = defaultdict(int)

    # Process each message group and determine paths
    for msg_ind, group in enumerate(message_groups):
        if "version" not in group or group["version"] != 1:
            if debug_output:
                print(f"Skipping message {msg_ind}: Invalid version")
            continue

        # Extract ensemble member information
        ensemble_member = None
        try:
            # Check various potential locations for ensemble info
            if ".zattrs" in group["refs"]:
                root_attrs = json.loads(group["refs"][".zattrs"])
                if "ensemble_member" in root_attrs:
                    ensemble_member = root_attrs["ensemble_member"]

            # Look for number variable which typically holds ensemble number
            if ensemble_member is None:
                for key in group["refs"]:
                    if key == "number/0" or key.endswith("/number/0"):
                        val = group["refs"][key]
                        if isinstance(val, str):
                            try:
                                arr = np.frombuffer(val.encode('latin1'),
                                                    dtype=np.int64)
                                if len(arr) == 1:
                                    ensemble_member = int(arr[0])
                                    break
                            except:
                                pass
        except Exception as e:
            if debug_output:
                print(
                    f"Warning: Error extracting ensemble information for msg {msg_ind}: {e}"
                )

        # Try to extract coordinates from the root attributes
        try:
            gattrs = json.loads(group["refs"][".zattrs"])
            coordinates = gattrs["coordinates"].split(" ")
        except Exception as e:
            if debug_output:
                print(
                    f"Warning: Issue with attributes for message {msg_ind}: {e}"
                )
            continue

        # Find the data variable
        vname = None
        for key in group["refs"]:
            name = key.split("/")[0]
            if name not in [".zattrs", ".zgroup"] and name not in coordinates:
                vname = name
                break

        if vname is None or vname == "unknown":
            if debug_output:
                print(
                    f"Warning: No valid data variable found for message {msg_ind}"
                )
            continue

        # Extract attributes for this variable
        try:
            dattrs = json.loads(group["refs"][f"{vname}/.zattrs"])
        except Exception as e:
            if debug_output:
                print(
                    f"Warning: Issue with variable attributes for {vname} in message {msg_ind}: {e}"
                )
            continue

        # Build path based on filter attributes
        gfilters = {}
        for key in filters:
            attr_val = dattrs.get(f"GRIB_{key}")
            if attr_val and attr_val != "unknown":
                gfilters[key] = attr_val

        # Start with variable name
        path_parts = [vname]

        # Add filter values to path
        for key, value in gfilters.items():
            if value:
                path_parts.append(value)

        # The base path excludes ensemble information
        base_path = "/".join(path_parts)

        # Add group to aggregations
        group_copy = copy.deepcopy(group)
        if ensemble_member is not None:
            group_copy["ensemble_member"] = ensemble_member

        aggregations[base_path].append(group_copy)
        path_counts[base_path] += 1

        # Track ensemble dimension
        if ensemble_member is not None:
            ensemble_dimensions[base_path].add(ensemble_member)

        # Track level information
        for key, entry in group["refs"].items():
            name = key.split("/")[0]
            if name == gfilters.get("typeOfLevel") and key.endswith("0"):
                if isinstance(entry, list):
                    entry = tuple(entry)
                level_dimensions[base_path].add(entry)

    # Print diagnostics for paths if debug is enabled
    if debug_output:
        print(
            f"Found {len(aggregations)} unique paths from {len(message_groups)} messages"
        )
        for path, groups in sorted(aggregations.items(),
                                   key=lambda x: len(x[1]),
                                   reverse=True):
            ensemble_count = len(ensemble_dimensions.get(path, set()))
            level_count = len(level_dimensions.get(path, set()))
            print(
                f"  {path}: {len(groups)} groups, {ensemble_count} ensemble members, {level_count} levels"
            )

    # Process each path with MultiZarrToZarr and ensure proper hierarchical structure
    for path, groups in aggregations.items():
        # Build groups for each level in the hierarchy
        path_parts = path.split("/")
        current_path = ""
        for i, part in enumerate(path_parts):
            prev_path = current_path

            if current_path:
                current_path = f"{current_path}/{part}"
            else:
                current_path = part

            # Add .zgroup for this level if not already present
            if f"{current_path}/.zgroup" not in zarr_store:
                zarr_store[f"{current_path}/.zgroup"] = json.dumps(
                    {'zarr_format': 2})

            # Add .zattrs for this level
            if f"{current_path}/.zattrs" not in zarr_store:
                # Add appropriate attributes based on the level
                attrs = {}

                # Add filter-specific attributes
                if i == 1 and len(path_parts) > 1:  # stepType level
                    attrs["stepType"] = path_parts[i]
                if i == 2 and len(path_parts) > 2:  # typeOfLevel level
                    attrs["typeOfLevel"] = path_parts[i]

                zarr_store[f"{current_path}/.zattrs"] = json.dumps(attrs)

        # Get dimensions for this path
        catdims = ["time", "step"]  # Always concatenate time and step
        idims = ["longitude",
                 "latitude"]  # Latitude and longitude are always identical

        # Handle level dimensions
        level_count = len(level_dimensions.get(path, set()))
        level_name = path_parts[-1] if len(path_parts) > 0 else None

        if level_count == 1:
            # Single level - treat as identical dimension
            if level_name and level_name not in idims:
                idims.append(level_name)
        elif level_count > 1:
            # Multiple levels - treat as concat dimension
            if level_name and level_name not in catdims:
                catdims.append(level_name)

        # Handle ensemble dimension
        ensemble_count = len(ensemble_dimensions.get(path, set()))
        if ensemble_count > 1 and "number" not in catdims:
            catdims.append("number")
            # Sort groups by ensemble number for consistent processing
            groups.sort(key=lambda g: g.get("ensemble_member", 0))

        if debug_output:
            print(
                f"Processing {path} with concat_dims={catdims}, identical_dims={idims}"
            )

        try:
            # Create aggregation
            mzz = MultiZarrToZarr(
                groups,
                remote_options=remote_options,
                concat_dims=catdims,
                identical_dims=idims,
            )

            # Get result and store references
            group_result = mzz.translate()

            # Add each reference with proper path prefix
            for key, value in group_result["refs"].items():
                if key == ".zattrs" or key == ".zgroup":
                    # Don't overwrite existing group metadata
                    if f"{path}/{key}" not in zarr_store:
                        zarr_store[f"{path}/{key}"] = value
                else:
                    # Data or other references
                    zarr_store[f"{path}/{key}"] = value

        except Exception as e:
            if debug_output:
                print(f"Error processing path {path}: {e}")
                import traceback
                traceback.print_exc()

    # Convert all byte values to strings for compatibility
    zarr_store = {
        key: (val.decode('utf-8') if isinstance(val, bytes) else val)
        for key, val in zarr_store.items()
    }

    return {"refs": zarr_store, "version": 1}


def analyze_grib_tree_output(original_tree, ensembe_tree):
    """
    Analyze and compare outputs from different grib_tree functions
    """

    # Count references by path prefix
    def count_by_prefix(refs_dict):
        prefix_counts = {}
        for key in refs_dict:
            # Extract first part of the path
            parts = key.split('/')
            if len(parts) > 0:
                prefix = parts[0]
                if prefix not in prefix_counts:
                    prefix_counts[prefix] = 0
                prefix_counts[prefix] += 1

        return prefix_counts

    # Count references by group level
    def count_by_level(refs_dict):
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for key in refs_dict:
            # Count levels in the path
            level = key.count('/')
            if level in level_counts:
                level_counts[level] += 1
            else:
                level_counts[5] += 1  # Group anything deeper than level 4

        return level_counts

    # Look for ensemble-related entries
    def find_ensemble_refs(refs_dict):
        ensemble_refs = []
        for key in refs_dict:
            if 'number' in key or 'ensemble' in key:
                ensemble_refs.append(key)

        return ensemble_refs

    # Analyze original tree
    orig_prefix_counts = count_by_prefix(original_tree['refs'])
    orig_level_counts = count_by_level(original_tree['refs'])
    orig_ensemble_refs = find_ensemble_refs(original_tree['refs'])

    # Analyze ensemble tree
    ens_prefix_counts = count_by_prefix(ensembe_tree['refs'])
    ens_level_counts = count_by_level(ensembe_tree['refs'])
    ens_ensemble_refs = find_ensemble_refs(ensembe_tree['refs'])

    # Print analysis
    print("=== ORIGINAL TREE ANALYSIS ===")
    print(f"Total references: {len(original_tree['refs'])}")
    print("\nReferences by variable prefix:")
    for prefix, count in sorted(orig_prefix_counts.items(),
                                key=lambda x: x[1],
                                reverse=True):
        print(f"  {prefix}: {count}")

    print("\nReferences by path depth:")
    for level, count in orig_level_counts.items():
        print(f"  Level {level}: {count}")

    print(f"\nEnsemble-related references: {len(orig_ensemble_refs)}")
    if orig_ensemble_refs:
        print("  Examples:")
        for ref in orig_ensemble_refs[:5]:  # Show up to 5 examples
            print(f"    {ref}")

    print("\n=== ENSEMBLE TREE ANALYSIS ===")
    print(f"Total references: {len(ensembe_tree['refs'])}")
    print("\nReferences by variable prefix:")
    for prefix, count in sorted(ens_prefix_counts.items(),
                                key=lambda x: x[1],
                                reverse=True):
        print(f"  {prefix}: {count}")

    print("\nReferences by path depth:")
    for level, count in ens_level_counts.items():
        print(f"  Level {level}: {count}")

    print(f"\nEnsemble-related references: {len(ens_ensemble_refs)}")
    if ens_ensemble_refs:
        print("  Examples:")
        for ref in ens_ensemble_refs[:5]:  # Show up to 5 examples
            print(f"    {ref}")

    # Compare structure of a sample variable
    print("\n=== SAMPLE VARIABLE COMPARISON ===")
    # Find a common variable prefix
    common_prefixes = set(orig_prefix_counts.keys()) & set(
        ens_prefix_counts.keys())
    if common_prefixes:
        sample_var = next(iter(common_prefixes))
        print(f"Sample variable: {sample_var}")

        # Get all paths for this variable
        orig_var_paths = [
            p for p in original_tree['refs'] if p.startswith(f"{sample_var}/")
        ]
        ens_var_paths = [
            p for p in ensembe_tree['refs'] if p.startswith(f"{sample_var}/")
        ]

        print(f"Original tree paths: {len(orig_var_paths)}")
        for path in sorted(orig_var_paths)[:5]:  # Show up to 5 examples
            print(f"  {path}")

        print(f"Ensemble tree paths: {len(ens_var_paths)}")
        for path in sorted(ens_var_paths)[:5]:  # Show up to 5 examples
            print(f"  {path}")

    return {
        "original": {
            "total": len(original_tree['refs']),
            "by_prefix": orig_prefix_counts,
            "by_level": orig_level_counts,
            "ensemble_refs": len(orig_ensemble_refs)
        },
        "ensemble": {
            "total": len(ensembe_tree['refs']),
            "by_prefix": ens_prefix_counts,
            "by_level": ens_level_counts,
            "ensemble_refs": len(ens_ensemble_refs)
        }
    }


def zstore_dict_to_df(gfs_store: Dict) -> pd.DataFrame:
    """Convert zarr store dictionary to DataFrame for parquet storage."""
    refs = gfs_store.get("refs", {})
    records = []

    for key, value in refs.items():
        if isinstance(value, dict):
            record = {"key": key, "value": json.dumps(value)}
        elif isinstance(value, list):
            if len(value) == 3:
                record = {
                    "key": key,
                    "path": value[0],
                    "offset": value[1],
                    "size": value[2]
                }
            else:
                record = {"key": key, "value": json.dumps(value)}
        else:
            if isinstance(value, (int, float, np.integer, np.floating)):
                record = {"key": key, "value": str(value)}
            else:
                record = {
                    "key":
                    key,
                    "value":
                    value.encode('utf-8').decode('utf-8') if isinstance(
                        value, bytes) else str(value)
                }
        records.append(record)

    return pd.DataFrame(records)


def create_parquet_file(zstore: dict, output_parquet_file: str):
    """Save zarr store dictionary as a parquet file."""
    gfs_store = dict(refs=zstore, version=1)
    zstore_df = zstore_dict_to_df(gfs_store)
    zstore_df.to_parquet(output_parquet_file)
    print(f"Parquet file saved to {output_parquet_file}")


def save_datatree_structure(dt, output_file: str):
    """Save datatree structure information to a JSON file for inspection."""
    structure = {}

    def extract_node_info(node, path=""):
        node_info = {
            "dims":
            dict(node.dims) if hasattr(node, 'dims') else {},
            "coords":
            list(node.coords.keys()) if hasattr(node, 'coords') else [],
            "data_vars":
            list(node.data_vars.keys()) if hasattr(node, 'data_vars') else [],
            "attrs":
            dict(node.attrs) if hasattr(node, 'attrs') else {}
        }
        return node_info

    # Extract root info
    structure["root"] = extract_node_info(dt)

    # Extract info for each group/variable
    for key in dt.keys():
        structure[key] = extract_node_info(dt[key])

    with open(output_file, 'w') as f:
        json.dump(structure, f, indent=2)
    print(f"DataTree structure saved to {output_file}")


# # ===== Final execution starts here =====

# script_start = log_checkpoint("Starting ECMWF ensemble processing script")

# date_str = '20250628'
# run='18'
# ecmwf_files = [
#     f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/{date_str}{run}0000-0h-enfo-ef.grib2",
#     f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/{date_str}{run}0000-3h-enfo-ef.grib2"
# ]

# log_checkpoint(
#     f"Processing {len(ecmwf_files)} ECMWF files for date {date_str}")

# # Process files and collect groups
# file_processing_start = log_checkpoint("Starting file processing")
# all_groups = []
# for i, eurl in enumerate(ecmwf_files, 1):
#     try:
#         file_start = log_checkpoint(
#             f"Processing file {i}/{len(ecmwf_files)}: {eurl.split('/')[-1]}")
#         groups, idx_mapping = ecmwf_filter_scan_grib(eurl)
#         all_groups.extend(groups)
#         log_checkpoint(f"File {i} completed, found {len(groups)} groups",
#                        file_start)
#     except Exception as e:
#         print(f"Error processing {eurl}: {e}")
#         import traceback
#         traceback.print_exc()

# log_checkpoint(f"File processing completed. Total groups: {len(all_groups)}",
#                file_processing_start)

# if not all_groups:
#     raise ValueError("No valid groups were found")

# # Skip the problematic grib_tree call for now - focus on fixed_ensemble_grib_tree
# log_checkpoint(
#     "Skipping standard grib_tree (credentials issue) - proceeding with fixed_ensemble_grib_tree"
# )

# # Create output directory for results
# #output_dir = f"ecmwf_test_results_{date_str}"
# output_dir = f"e_{date_str}_{run}"
# os.makedirs(output_dir, exist_ok=True)
# log_checkpoint(f"Created output directory: {output_dir}")

# # Build ensemble tree with proper remote options
# ensemble_start = log_checkpoint("Starting fixed_ensemble_grib_tree processing")
# remote_options = {"anon": True}  # Anonymous S3 access
# ensemble_tree = fixed_ensemble_grib_tree(all_groups,
#                                          remote_options=remote_options,
#                                          debug_output=True)
# log_checkpoint(
#     f"Ensemble tree built with {len(ensemble_tree['refs'])} references",
#     ensemble_start)

# # Save the raw ensemble tree
# save_start = log_checkpoint("Saving raw ensemble tree to JSON")
# with open(f"{output_dir}/ensemble_tree_raw.json", 'w') as f:
#     # Convert to serializable format
#     serializable_tree = copy.deepcopy(ensemble_tree)
#     json.dump(serializable_tree, f, indent=2)
# log_checkpoint(
#     f"Saved raw ensemble tree to {output_dir}/ensemble_tree_raw.json",
#     save_start)

# # Create deflated store for parquet (following ECMWF pattern)
# deflate_start = log_checkpoint("Creating deflated store for parquet")
# deflated_ecmwf_grib_tree_store = copy.deepcopy(ensemble_tree)
# strip_datavar_chunks(deflated_ecmwf_grib_tree_store)
# log_checkpoint("Deflated store created", deflate_start)

# # Save deflated store as parquet
# parquet_start = log_checkpoint("Saving deflated store as parquet file")
# parquet_file = f"{output_dir}/ecmwf_{date_str}_00z_ensemble.parquet"
# create_parquet_file(deflated_ecmwf_grib_tree_store['refs'], parquet_file)
# log_checkpoint(f"Parquet file saved: {parquet_file}", parquet_start)

# # Check the references directly
# print(f"\nTotal refs in ensemble tree: {len(ensemble_tree['refs'])}")
# print(
#     f"Total refs in deflated store: {len(deflated_ecmwf_grib_tree_store['refs'])}"
# )

# # Look at structure - should have proper hierarchy
# print("\nSample keys from ensemble tree:")
# print([key for key in ensemble_tree['refs'].keys()
#        if key.count('/') <= 1][:10])

# # Open with datatree
# datatree_start = log_checkpoint("Opening with xarray datatree")
# try:
#     egfs_dt = xr.open_datatree(fsspec.filesystem(
#         "reference", fo=ensemble_tree).get_mapper(""),
#                                engine="zarr",
#                                consolidated=False)

#     log_checkpoint("DataTree opened successfully", datatree_start)

#     # Save datatree structure
#     structure_start = log_checkpoint("Saving DataTree structure analysis")
#     save_datatree_structure(egfs_dt, f"{output_dir}/datatree_structure.json")
#     log_checkpoint("DataTree structure saved", structure_start)

#     # Check for variables
#     print(f"\nDataTree keys: {list(egfs_dt.keys())}")

#     # Try accessing a variable
#     if 't2m' in egfs_dt.keys():
#         var_start = log_checkpoint("Analyzing t2m variable")
#         var_node = egfs_dt['t2m']
#         print(f"\nt2m dimensions: {var_node.dims}")

#         # Save sample data info
#         var_info = {
#             "variable": "t2m",
#             "dims": dict(var_node.dims),
#             "coords": list(var_node.coords.keys()),
#             "attrs": dict(var_node.attrs)
#         }
#         with open(f"{output_dir}/sample_variable_info.json", 'w') as f:
#             json.dump(var_info, f, indent=2)
#         log_checkpoint("Variable analysis completed", var_start)

#     # Save the egfs_dt object using pickle for later analysis
#     pickle_start = log_checkpoint("Saving DataTree object as pickle")
#     with open(f"{output_dir}/egfs_dt.pkl", 'wb') as f:
#         pickle.dump(egfs_dt, f)
#     log_checkpoint(f"DataTree object saved to {output_dir}/egfs_dt.pkl",
#                    pickle_start)

# except Exception as e:
#     log_checkpoint(f"Error opening with datatree: {e}")
#     import traceback
#     traceback.print_exc()

# # Also save intermediate results for debugging
# summary_start = log_checkpoint("Creating processing summary")
# intermediate_results = {
#     "date_str":
#     date_str,
#     "ecmwf_files":
#     ecmwf_files,
#     "num_groups":
#     len(all_groups),
#     "ensemble_tree_keys_count":
#     len(ensemble_tree['refs']),
#     "deflated_store_keys_count":
#     len(deflated_ecmwf_grib_tree_store['refs']),
#     "sample_keys":
#     [key for key in ensemble_tree['refs'].keys() if key.count('/') <= 1][:20]
# }

# with open(f"{output_dir}/processing_summary.json", 'w') as f:
#     json.dump(intermediate_results, f, indent=2)
# log_checkpoint("Processing summary saved", summary_start)

# # Final timing
# total_time = time.time() - script_start
# log_checkpoint(f"=== Processing complete === (Total time: {total_time:.2f}s)")
# print(f"All results saved to: {output_dir}/")
# print(f"\nKey files created:")
# print(f"  - {parquet_file} (main output for further processing)")
# print(f"  - {output_dir}/ensemble_tree_raw.json (raw tree structure)")
# print(f"  - {output_dir}/datatree_structure.json (DataTree structure)")
# print(f"  - {output_dir}/egfs_dt.pkl (pickled DataTree object)")
# print(f"  - {output_dir}/processing_summary.json (processing summary)")
# print(f"\nTotal processing time: {total_time:.2f} seconds")
