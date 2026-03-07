#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "xarray",
#     "netCDF4",
#     "earthkit-regrid",
#     "python-dotenv",
#     "google-cloud-storage",
# ]
# ///
"""
ERA5T EDA Input States Creation from CEDA Archive

Downloads ERA5T Ensemble of Data Assimilations (EDA) data from CEDA and
creates input states for AIFSgaia ensemble forecasting.

Data source: https://data.ceda.ac.uk/badc/ecmwf-era5t/data/
Uses wget with Bearer token authentication per:
  https://help.ceda.ac.uk/article/5191-downloading-multiple-files-with-wget

Strategy:
  - Surface fields (10u, 10v, 2t, msl, tcwv): per-member from enda/an_sfc
  - Pressure level fields (t, q, z, u, v x 13 levels): deterministic from
    oper/an_ml, interpolated from 137 model levels to pressure levels using
    ECMWF L137 hybrid coefficients. Shared identically across all 10 members.
  - Two consecutive states: t and t-24h

Usage:
  uv run ceda_era5t_pkl_input_aifsens.py
"""

import datetime
import os
import pickle
import subprocess
import tempfile
import time

import numpy as np
import xarray as xr
import earthkit.regrid as ekr
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────────────

CEDA_ENDA_SFC_BASE = "https://data.ceda.ac.uk/badc/ecmwf-era5t/data/enda/an_sfc"
CEDA_OPER_ML_BASE = "https://data.ceda.ac.uk/badc/ecmwf-era5t/data/oper/an_ml"
CEDA_TOKEN = os.environ.get("ceda_token", "")

PARAM_SFC_EDA = ["10u", "10v", "2t", "msl", "tcwv"]
PARAM_PL = ["t", "q", "z", "u", "v"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

ENSEMBLE_MEMBERS = list(range(0, 10))
TIME_OFFSET_HOURS = 24

OUTPUT_DIR = "test_input_states"
SAVE_STATES = True

GCS_BUCKET = "aifs-aiquest-us-20251127"
GCS_SERVICE_ACCOUNT_KEY = "coiled-data.json"
UPLOAD_TO_GCS = True
CLEANUP_LOCAL_FILES = True

DOWNLOAD_DIR = tempfile.mkdtemp(prefix="ceda_era5t_")

# ─── ECMWF L137 hybrid coefficients ─────────────────────────────────────────
# Half-level coefficients: p_half(k) = a(k) + b(k) * sp  for k=0..137
# Full-level pressure: p_full(k) = (p_half(k-1) + p_half(k)) / 2  for k=1..137
# Source: https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions

A_HALF = np.array([
    0.000000, 2.000365, 3.102241, 4.666084, 6.827977, 9.746966, 13.605424,
    18.608931, 24.985718, 32.985710, 42.879242, 54.955463, 69.520576,
    86.895882, 107.415741, 131.425507, 159.279404, 191.338562, 227.968948,
    269.539581, 316.420746, 368.982361, 427.592499, 492.616028, 564.413452,
    643.339905, 729.744141, 823.967834, 926.344910, 1037.201172, 1156.853638,
    1285.610352, 1423.770142, 1571.622925, 1729.448975, 1897.519287,
    2076.095947, 2265.431641, 2465.770508, 2677.348145, 2900.391357,
    3135.119385, 3381.743652, 3640.468262, 3911.490479, 4194.930664,
    4490.817383, 4799.149414, 5119.895020, 5452.990723, 5798.344727,
    6156.074219, 6526.946777, 6911.870605, 7311.869141, 7727.412109,
    8159.354004, 8608.525391, 9076.400391, 9562.682617, 10065.978516,
    10584.631836, 11116.662109, 11660.067383, 12211.547852, 12766.873047,
    13324.668945, 13881.331055, 14432.139648, 14975.615234, 15508.256836,
    16026.115234, 16527.322266, 17008.789063, 17467.613281, 17901.621094,
    18308.433594, 18685.718750, 19031.289063, 19343.511719, 19620.042969,
    19859.390625, 20059.931641, 20219.664063, 20337.863281, 20412.308594,
    20442.078125, 20425.718750, 20361.816406, 20249.511719, 20087.085938,
    19874.025391, 19608.572266, 19290.226563, 18917.460938, 18489.707031,
    18006.925781, 17471.839844, 16888.687500, 16262.046875, 15596.695313,
    14898.453125, 14173.324219, 13427.769531, 12668.257813, 11901.339844,
    11133.304688, 10370.175781, 9617.515625, 8880.453125, 8163.375000,
    7470.343750, 6804.421875, 6168.531250, 5564.382813, 4993.796875,
    4457.375000, 3955.960938, 3489.234375, 3057.265625, 2659.140625,
    2294.242188, 1961.500000, 1659.476563, 1387.546875, 1143.250000,
    926.507813, 734.992188, 568.062500, 424.414063, 302.476563, 202.484375,
    122.101563, 62.781250, 22.835938, 3.757813, 0.000000, 0.000000,
])

B_HALF = np.array([
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000007,
    0.000024, 0.000059, 0.000112, 0.000199, 0.000340, 0.000562, 0.000890,
    0.001353, 0.001992, 0.002857, 0.003971, 0.005378, 0.007133, 0.009261,
    0.011806, 0.014816, 0.018318, 0.022355, 0.026964, 0.032176, 0.038026,
    0.044548, 0.051773, 0.059728, 0.068448, 0.077958, 0.088286, 0.099462,
    0.111505, 0.124448, 0.138313, 0.153125, 0.168910, 0.185689, 0.203491,
    0.222333, 0.242244, 0.263242, 0.285354, 0.308598, 0.332939, 0.358254,
    0.384363, 0.411125, 0.438391, 0.466003, 0.493800, 0.521619, 0.549301,
    0.576692, 0.603648, 0.630036, 0.655736, 0.680643, 0.704669, 0.727739,
    0.749797, 0.770798, 0.790717, 0.809536, 0.827256, 0.843881, 0.859432,
    0.873929, 0.887408, 0.899900, 0.911448, 0.922096, 0.931881, 0.940860,
    0.949064, 0.956550, 0.963352, 0.969513, 0.975078, 0.980072, 0.984542,
    0.988500, 0.991984, 0.995003, 0.997630, 1.000000,
])


# ─── Download helpers ────────────────────────────────────────────────────────

def build_ceda_url(base_url, dt, member=None, param=None):
    """Build CEDA file URL.

    enda: ecmwf-era5t_enda_an_sfc_YYYYMMDDHHMM.memN.PARAM.nc
    oper: ecmwf-era5t_oper_an_ml_YYYYMMDDHHMM.PARAM.nc
    """
    datestr = dt.strftime("%Y/%m/%d")
    timestamp = dt.strftime("%Y%m%d%H%M")

    if member is not None:
        filename = f"ecmwf-era5t_enda_an_sfc_{timestamp}.mem{member}.{param}.nc"
        return f"{base_url}/{datestr}/{filename}"
    else:
        filename = f"ecmwf-era5t_oper_an_ml_{timestamp}.{param}.nc"
        return f"{base_url}/{datestr}/{filename}"


def wget_download(url, output_path, token):
    cmd = [
        "wget", "-q",
        "--header", f"Authorization: Bearer {token}",
        "-O", output_path,
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"wget failed for {url}: {result.stderr}")
    return output_path


def download_ceda_nc(dt, member=None, param=None, base_url=None):
    if base_url is None:
        base_url = CEDA_ENDA_SFC_BASE if member is not None else CEDA_OPER_ML_BASE

    url = build_ceda_url(base_url, dt, member=member, param=param)

    if member is not None:
        local_name = f"enda_sfc_{dt.strftime('%Y%m%d%H%M')}_mem{member}_{param}.nc"
    else:
        local_name = f"oper_ml_{dt.strftime('%Y%m%d%H%M')}_{param}.nc"

    local_path = os.path.join(DOWNLOAD_DIR, local_name)

    if os.path.exists(local_path):
        return local_path

    print(f"      Downloading {param}" +
          (f" mem{member}" if member is not None else "") +
          f" {dt.strftime('%Y-%m-%d %H:%M')}...")
    wget_download(url, local_path, CEDA_TOKEN)
    return local_path


# ─── Data reading ────────────────────────────────────────────────────────────

def read_sfc_nc(nc_path):
    """Read a CEDA surface NetCDF file -> 1D N320 array."""
    ds = xr.open_dataset(nc_path)
    data_vars = [v for v in ds.data_vars if v not in ('latitude', 'longitude', 'time')]
    values = ds[data_vars[0]].values.squeeze()
    assert values.ndim == 2, f"Expected 2D, got {values.shape}"
    # CEDA is already 0-360 longitude
    values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
    ds.close()
    return values


def get_enda_sfc_data(date, param, member):
    """Get EDA surface field for t-24h and t -> dict {param: (2, N320)}."""
    timestamps = [date - datetime.timedelta(hours=TIME_OFFSET_HOURS), date]
    values_list = []
    for dt in timestamps:
        nc_path = download_ceda_nc(dt, member=member, param=param,
                                   base_url=CEDA_ENDA_SFC_BASE)
        values_list.append(read_sfc_nc(nc_path))
    return {param: np.stack(values_list)}


# ─── Model level -> Pressure level interpolation ────────────────────────────

def compute_pressure_on_model_levels(lnsp_2d):
    """Compute full-level pressures from lnsp using L137 hybrid coefficients.

    Args:
        lnsp_2d: log surface pressure, shape (lat, lon)

    Returns:
        p_full: pressure at each of 137 model levels, shape (137, lat, lon) in Pa
    """
    sp = np.exp(lnsp_2d)  # (lat, lon)
    # Half-level pressures: p_half(k) = a(k) + b(k) * sp, shape (138, lat, lon)
    p_half = A_HALF[:, None, None] + B_HALF[:, None, None] * sp[None, :, :]
    # Full-level pressures: average of bounding half-levels
    p_full = (p_half[:-1] + p_half[1:]) / 2  # (137, lat, lon)
    return p_full


def interpolate_ml_to_pl_linear(ml_data, p_full, target_hpa):
    """Linearly interpolate a model-level field to a target pressure level.

    Interpolation is done in log-pressure space for physical consistency.

    Args:
        ml_data: shape (137, lat, lon) - field on model levels
        p_full: shape (137, lat, lon) - pressure at each model level in Pa
        target_hpa: target pressure in hPa

    Returns:
        interpolated field, shape (lat, lon)
    """
    target_pa = target_hpa * 100.0
    nlat, nlon = ml_data.shape[1], ml_data.shape[2]
    result = np.empty((nlat, nlon), dtype=ml_data.dtype)

    log_p = np.log(p_full)  # (137, lat, lon)
    log_target = np.log(target_pa)

    # For each grid point, find the two model levels bracketing the target
    # Pressure increases with level index (level 1=top, 137=bottom)
    # Find where log_p crosses log_target
    # below_idx[j,k] = first level where p >= target (level below)
    above_target = log_p >= log_target  # (137, lat, lon)

    # Index of first level at or below target pressure
    # argmax on bool gives first True
    below_idx = np.argmax(above_target, axis=0)  # (lat, lon)

    # Clamp indices
    above_idx = np.clip(below_idx - 1, 0, 136)
    below_idx = np.clip(below_idx, 0, 136)

    # Gather values at bracketing levels
    lat_idx, lon_idx = np.meshgrid(np.arange(nlat), np.arange(nlon), indexing='ij')

    log_p_above = log_p[above_idx, lat_idx, lon_idx]
    log_p_below = log_p[below_idx, lat_idx, lon_idx]
    val_above = ml_data[above_idx, lat_idx, lon_idx]
    val_below = ml_data[below_idx, lat_idx, lon_idx]

    # Linear interpolation in log-p space
    denom = log_p_below - log_p_above
    # Avoid division by zero where above_idx == below_idx
    safe_denom = np.where(denom == 0, 1.0, denom)
    weight = (log_target - log_p_above) / safe_denom
    weight = np.clip(weight, 0.0, 1.0)

    result = val_above + weight * (val_below - val_above)
    return result


def compute_geopotential_on_ml(t_ml, q_ml, z_sfc, p_full, p_half):
    """Compute geopotential at full model levels using the hypsometric equation.

    Integrates upward from the surface using virtual temperature.
    Phi(k) = Phi(k+1) + Rd * Tv * ln(p_full(k+1) / p_full(k))

    Args:
        t_ml: temperature on model levels, shape (137, lat, lon)
        q_ml: specific humidity on model levels, shape (137, lat, lon)
        z_sfc: surface geopotential, shape (lat, lon) in m^2/s^2
        p_full: full-level pressures, shape (137, lat, lon) in Pa
        p_half: half-level pressures, shape (138, lat, lon) in Pa

    Returns:
        geopotential on model levels, shape (137, lat, lon) in m^2/s^2
    """
    Rd = 287.058  # J/(kg*K), gas constant for dry air
    eps = 0.621971  # Rd/Rv ratio

    # Virtual temperature: Tv = T * (1 + ((1/eps - 1) * q))
    tv = t_ml * (1.0 + (1.0 / eps - 1.0) * q_ml)

    n_levels = t_ml.shape[0]
    geo = np.zeros_like(t_ml)

    # Start from the bottom (level 137, index 136)
    # Geopotential at the bottom half-level is the surface geopotential
    # geo_half(137) = z_sfc
    # geo_full(137) = z_sfc + Rd * Tv(137) * ln(p_half(137) / p_full(137))
    geo[n_levels - 1] = z_sfc + Rd * tv[n_levels - 1] * np.log(
        p_half[n_levels] / p_full[n_levels - 1])

    # Integrate upward: k = 136 down to 0 (level 136 to level 1)
    for k in range(n_levels - 2, -1, -1):
        # Geopotential at half-level k+1 (between full levels k and k+1)
        geo_half_below = geo[k + 1] + Rd * tv[k + 1] * np.log(
            p_full[k + 1] / p_half[k + 1])
        # Geopotential at full level k
        geo[k] = geo_half_below + Rd * tv[k] * np.log(
            p_half[k + 1] / p_full[k])

    return geo


def get_pl_fields_from_ml(date):
    """Download oper/an_ml data and interpolate to pressure levels.

    Downloads t, q, u, v on 137 model levels plus lnsp and surface z.
    Computes geopotential on model levels via hypsometric equation.
    Interpolates all fields to target pressure levels.

    Returns dict of {param_level: (2, N320)} for all PL fields.
    """
    timestamps = [date - datetime.timedelta(hours=TIME_OFFSET_HOURS), date]
    fields = {}

    # Download all needed files
    print("    Downloading lnsp + z (surface)...")
    lnsp_paths, z_sfc_paths = [], []
    for dt in timestamps:
        lnsp_paths.append(download_ceda_nc(dt, param="lnsp", base_url=CEDA_OPER_ML_BASE))
        z_sfc_paths.append(download_ceda_nc(dt, param="z", base_url=CEDA_OPER_ML_BASE))

    # Download model-level params: t, q, u, v (z is computed, not directly available)
    ml_params = ["t", "q", "u", "v"]
    ml_paths = {}
    for param in ml_params:
        print(f"    Downloading {param} (137 model levels)...")
        ml_paths[param] = []
        for dt in timestamps:
            ml_paths[param].append(
                download_ceda_nc(dt, param=param, base_url=CEDA_OPER_ML_BASE))

    # Process each timestep
    for level in LEVELS:
        print(f"    Interpolating to {level} hPa...")

    # Actually, process per-timestep to reuse loaded data efficiently
    print("    Interpolating all fields to pressure levels...")
    # Initialize: fields[param_level] will collect 2 timestep arrays
    field_arrays = {
        f"{'z' if p == 'z' else p}_{lev}": []
        for p in PARAM_PL for lev in LEVELS
    }

    for i, dt in enumerate(timestamps):
        ts_label = "t-24h" if i == 0 else "t"
        print(f"      Processing timestep {ts_label} ({dt})...")

        # Read lnsp -> surface pressure
        ds_lnsp = xr.open_dataset(lnsp_paths[i])
        lnsp_var = [v for v in ds_lnsp.data_vars
                    if v not in ('latitude', 'longitude', 'time')][0]
        lnsp_2d = ds_lnsp[lnsp_var].values.squeeze()
        ds_lnsp.close()

        sp = np.exp(lnsp_2d)
        p_half = A_HALF[:, None, None] + B_HALF[:, None, None] * sp[None, :, :]
        p_full = (p_half[:-1] + p_half[1:]) / 2

        # Read surface geopotential
        ds_z = xr.open_dataset(z_sfc_paths[i])
        z_var = [v for v in ds_z.data_vars
                 if v not in ('latitude', 'longitude', 'time')][0]
        z_sfc = ds_z[z_var].values.squeeze()  # (lat, lon) in m^2/s^2
        ds_z.close()

        # Read model-level t and q (needed for geopotential computation)
        ml_data = {}
        for param in ml_params:
            ds = xr.open_dataset(ml_paths[param][i])
            var_name = [v for v in ds.data_vars
                        if v not in ('latitude', 'longitude', 'time', 'level')][0]
            ml_data[param] = ds[var_name].values.squeeze()  # (137, lat, lon)
            ds.close()

        # Compute geopotential on model levels
        ml_data["z"] = compute_geopotential_on_ml(
            ml_data["t"], ml_data["q"], z_sfc, p_full, p_half)

        # Interpolate each param to each target pressure level
        for param in PARAM_PL:
            for level in LEVELS:
                values_2d = interpolate_ml_to_pl_linear(
                    ml_data[param], p_full, level)
                values_1d = ekr.interpolate(
                    values_2d, {"grid": (0.25, 0.25)}, {"grid": "N320"})
                field_name = f"z_{level}" if param == "z" else f"{param}_{level}"
                field_arrays[field_name].append(values_1d)

    # Stack timesteps
    for name, arrs in field_arrays.items():
        fields[name] = np.stack(arrs)  # (2, N320)

    return fields


# ─── Input state creation ────────────────────────────────────────────────────

def create_input_states(date):
    """Create input states for all EDA members.

    1. Download and interpolate PL fields once (deterministic, shared)
    2. For each member, download surface fields and combine with shared PL fields
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Pressure level fields (deterministic, download once)")
    print("=" * 60)

    pl_start = time.time()
    pl_fields = get_pl_fields_from_ml(date)
    pl_elapsed = time.time() - pl_start

    print(f"\n  PL fields: {len(pl_fields)} fields in {pl_elapsed:.1f}s")
    for name, arr in list(pl_fields.items())[:3]:
        print(f"    {name}: shape={arr.shape}, range=[{arr.min():.2f}, {arr.max():.2f}]")
    print(f"    ... and {len(pl_fields) - 3} more")

    print("\n" + "=" * 60)
    print("PHASE 2: Surface fields per EDA member + combine with PL")
    print("=" * 60)

    successful_members = []
    failed_members = []
    processing_times = []

    upload_enabled = UPLOAD_TO_GCS
    if upload_enabled and not os.path.exists(GCS_SERVICE_ACCOUNT_KEY):
        print(f"GCS key not found: {GCS_SERVICE_ACCOUNT_KEY}, disabling upload")
        upload_enabled = False

    datestr = date.strftime("%Y%m%d_%H%M")

    for i, member in enumerate(ENSEMBLE_MEMBERS):
        print(f"\n--- Member {member} ({i+1}/{len(ENSEMBLE_MEMBERS)}) ---")

        try:
            start_time = time.time()

            # Per-member surface fields
            fields = {}
            for param in PARAM_SFC_EDA:
                print(f"    {param}...")
                fields.update(get_enda_sfc_data(date, param, member))

            # Attach shared PL fields (same for all members)
            fields.update(pl_fields)

            input_state = dict(date=date, fields=fields)

            # Verify
            expected_count = len(PARAM_SFC_EDA) + len(PARAM_PL) * len(LEVELS)
            actual_count = len(fields)
            print(f"  Fields: {actual_count}/{expected_count} "
                  f"({len(PARAM_SFC_EDA)} sfc + {len(pl_fields)} pl)")

            if actual_count == expected_count:
                print(f"  All {expected_count} fields present!")
                successful_members.append(member)
            else:
                missing = expected_count - actual_count
                print(f"  WARNING: {missing} fields missing")

            # Save pkl
            if SAVE_STATES:
                filename = f"{OUTPUT_DIR}/input_state_member_{member:03d}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(input_state, f)
                print(f"  Saved to {filename}")

                if upload_enabled:
                    gcs_blob_name = f"{datestr}/input/input_state_member_{member:03d}.pkl"
                    if upload_to_gcs(filename, GCS_BUCKET, gcs_blob_name,
                                     GCS_SERVICE_ACCOUNT_KEY):
                        if CLEANUP_LOCAL_FILES:
                            os.remove(filename)

            elapsed = time.time() - start_time
            processing_times.append(elapsed)
            remaining = (len(ENSEMBLE_MEMBERS) - (i + 1)) * (sum(processing_times) / len(processing_times))
            print(f"  Time: {elapsed:.1f}s | Remaining: {remaining/60:.1f} min")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed_members.append(member)

    return successful_members, failed_members, processing_times, pl_elapsed


def upload_to_gcs(local_file_path, gcs_bucket, gcs_blob_name, service_account_key):
    try:
        client = storage.Client.from_service_account_json(service_account_key)
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_file_path)
        print(f"    Uploaded to gs://{gcs_bucket}/{gcs_blob_name}")
        return True
    except Exception as e:
        print(f"    Upload failed: {e}")
        return False


def main():
    if not CEDA_TOKEN:
        print("ERROR: ceda_token not found in .env file")
        return

    DATE = datetime.datetime(2026, 2, 27, 0, 0)
    print(f"Date: {DATE}")
    print(f"Previous state: {DATE - datetime.timedelta(hours=TIME_OFFSET_HOURS)}")
    print(f"Members: {len(ENSEMBLE_MEMBERS)} (mem{ENSEMBLE_MEMBERS[0]}-mem{ENSEMBLE_MEMBERS[-1]})")
    print(f"Fields per member: {len(PARAM_SFC_EDA)} sfc + {len(PARAM_PL)*len(LEVELS)} pl = "
          f"{len(PARAM_SFC_EDA) + len(PARAM_PL)*len(LEVELS)} total")
    print(f"Download dir: {DOWNLOAD_DIR}")

    if SAVE_STATES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    successful, failed, times, pl_time = create_input_states(DATE)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"PL download + interpolation: {pl_time/60:.1f} min (done once)")
    print(f"Members: {len(successful)}/{len(ENSEMBLE_MEMBERS)} successful")
    if failed:
        print(f"Failed: {failed}")
    if times:
        print(f"Sfc per member: {sum(times)/len(times):.1f}s avg")
        print(f"Total: {(pl_time + sum(times))/60:.1f} min")
    print(f"\nDownloaded files in: {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
