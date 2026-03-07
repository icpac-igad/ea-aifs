#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "xarray",
#     "netCDF4",
#     "earthkit-data",
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
  - Constant forcing fields (lsm, z, slor, sdor): from ECMWF Open Data
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
import earthkit.data as ekd
import earthkit.regrid as ekr
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────────────

CEDA_ENDA_SFC_BASE = "https://data.ceda.ac.uk/badc/ecmwf-era5t/data/enda/an_sfc"
CEDA_OPER_ML_BASE = "https://data.ceda.ac.uk/badc/ecmwf-era5t/data/oper/an_ml"
CEDA_TOKEN = os.environ.get("ceda_token", "")

PARAM_SFC_EDA = ["10u", "10v", "2t", "msl", "tcwv"]
PARAM_SFC_CONST = ["lsm", "z", "slor", "sdor"]  # Static forcing fields from ECMWF Open Data
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


def get_constant_fields():
    """Get static forcing fields (lsm, z, slor, sdor) from ECMWF Open Data.

    These are time-invariant constants — downloaded once and shared across
    all members and timesteps. Stacked as (2, N320) with identical values
    in both timesteps to match the expected input shape.
    """
    print("    Downloading constant fields from ECMWF Open Data...")
    data = ekd.from_source("ecmwf-open-data", param=PARAM_SFC_CONST)

    fields = {}
    for f in data:
        values = f.to_numpy()
        assert values.shape == (721, 1440), f"Unexpected shape: {values.shape}"
        # ECMWF Open Data is -180 to 180, shift to 0-360
        values = np.roll(values, -values.shape[1] // 2, axis=1)
        values_1d = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
        # Stack same values for both timesteps (constants don't change)
        fields[f.metadata("param")] = np.stack([values_1d, values_1d])
        print(f"      {f.metadata('param')}: range=[{values_1d.min():.4f}, {values_1d.max():.4f}]")

    return fields


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


def compute_geopotential_on_target_levels(t_ml, q_ml, z_sfc, lnsp_2d, target_levels_hpa):
    """Compute geopotential at specific pressure levels only (memory-efficient).

    Instead of computing geopotential on all 137 levels, integrates upward
    and only stores results at the target pressure levels.

    Uses float32 throughout to reduce memory (~4x less than float64).

    Args:
        t_ml: temperature on model levels, shape (137, lat, lon)
        q_ml: specific humidity on model levels, shape (137, lat, lon)
        z_sfc: surface geopotential, shape (lat, lon) in m^2/s^2
        lnsp_2d: log surface pressure, shape (lat, lon)
        target_levels_hpa: list of target pressure levels in hPa

    Returns:
        dict of {level_hpa: geopotential_2d} for each target level
    """
    Rd = np.float32(287.058)
    eps = np.float32(0.621971)

    # Convert to float32
    t_ml = t_ml.astype(np.float32)
    q_ml = q_ml.astype(np.float32)
    z_sfc = z_sfc.astype(np.float32)

    sp = np.exp(lnsp_2d).astype(np.float32)  # (lat, lon)
    nlat, nlon = sp.shape

    # Compute half-level pressures only for levels we need
    # p_half(k) = a(k) + b(k) * sp
    # We integrate from bottom up, so compute on the fly

    target_pa = sorted([lev * 100.0 for lev in target_levels_hpa], reverse=True)
    results = {}

    # Virtual temperature
    tv = t_ml * (np.float32(1.0) + (np.float32(1.0) / eps - np.float32(1.0)) * q_ml)

    # Integrate from bottom (k=136) upward
    n_levels = 137

    # Compute half-level pressures for bottom two half-levels
    p_half_below = (np.float32(A_HALF[n_levels]) +
                    np.float32(B_HALF[n_levels]) * sp)  # k=137 half-level = surface
    p_half_above = (np.float32(A_HALF[n_levels - 1]) +
                    np.float32(B_HALF[n_levels - 1]) * sp)  # k=136

    p_full_k = (p_half_above + p_half_below) / np.float32(2.0)

    # Geopotential at bottom full level
    geo_k = z_sfc + Rd * tv[n_levels - 1] * np.log(p_half_below / p_full_k)

    # Check if any target level is at or below the bottom level
    for tp in list(target_pa):
        if tp >= p_full_k.mean():
            results[int(tp / 100)] = geo_k.copy()
            target_pa.remove(tp)

    if not target_pa:
        return results

    # Move upward level by level
    for k in range(n_levels - 2, -1, -1):
        # Half-level pressures
        p_half_below_new = p_half_above.copy()
        p_half_above = (np.float32(A_HALF[k]) +
                        np.float32(B_HALF[k]) * sp)

        p_full_prev = p_full_k.copy()
        geo_prev = geo_k.copy()

        p_full_k = (p_half_above + p_half_below_new) / np.float32(2.0)

        # Geopotential at half-level between k+1 and k
        geo_half = geo_prev + Rd * tv[k + 1] * np.log(p_full_prev / p_half_below_new)
        # Geopotential at full level k
        geo_k = geo_half + Rd * tv[k] * np.log(p_half_below_new / p_full_k)

        # Check if we've passed any target levels (interpolate in log-p)
        for tp in list(target_pa):
            # Target between this level and the one below?
            mean_p_k = float(p_full_k.mean())
            mean_p_prev = float(p_full_prev.mean())
            if mean_p_k <= tp <= mean_p_prev:
                # Linear interpolation in log-p space
                log_pk = np.log(p_full_k)
                log_pp = np.log(p_full_prev)
                log_t = np.float32(np.log(tp))
                denom = log_pp - log_pk
                safe_denom = np.where(denom == 0, np.float32(1.0), denom)
                w = (log_t - log_pk) / safe_denom
                w = np.clip(w, np.float32(0.0), np.float32(1.0))
                geo_interp = geo_k + w * (geo_prev - geo_k)
                results[int(tp / 100)] = geo_interp
                target_pa.remove(tp)

        if not target_pa:
            break

    return results


def _read_ml_var(nc_path):
    """Read a single model-level NetCDF file and return the data array."""
    ds = xr.open_dataset(nc_path)
    var_name = [v for v in ds.data_vars
                if v not in ('latitude', 'longitude', 'time', 'level')][0]
    data = ds[var_name].values.squeeze()
    ds.close()
    return data


def _read_sfc_var(nc_path):
    """Read a single surface NetCDF file and return the 2D data array."""
    ds = xr.open_dataset(nc_path)
    var_name = [v for v in ds.data_vars
                if v not in ('latitude', 'longitude', 'time')][0]
    data = ds[var_name].values.squeeze()
    ds.close()
    return data


def get_pl_fields_from_ml(date):
    """Download oper/an_ml data and interpolate to pressure levels.

    Memory-efficient: processes one parameter at a time, only loading t+q
    together when computing geopotential.

    Returns dict of {param_level: (2, N320)} for all PL fields.
    """
    timestamps = [date - datetime.timedelta(hours=TIME_OFFSET_HOURS), date]
    fields = {}

    # Download all files first (small files + large ML files)
    print("    Downloading lnsp + z (surface)...")
    lnsp_paths, z_sfc_paths = [], []
    for dt in timestamps:
        lnsp_paths.append(download_ceda_nc(dt, param="lnsp", base_url=CEDA_OPER_ML_BASE))
        z_sfc_paths.append(download_ceda_nc(dt, param="z", base_url=CEDA_OPER_ML_BASE))

    ml_params = ["t", "q", "u", "v"]
    ml_paths = {}
    for param in ml_params:
        print(f"    Downloading {param} (137 model levels)...")
        ml_paths[param] = []
        for dt in timestamps:
            ml_paths[param].append(
                download_ceda_nc(dt, param=param, base_url=CEDA_OPER_ML_BASE))

    # Process each timestep: compute geopotential + interpolate all params
    # Only one ML file loaded at a time to stay within ~8GB RAM
    print("    Processing timesteps...")

    # Collect results: field_name -> [ts0_1d, ts1_1d]
    field_arrays = {}

    for i in range(2):
        ts_label = "t-24h" if i == 0 else "t"
        print(f"      {ts_label}: loading lnsp...")

        lnsp_2d = _read_sfc_var(lnsp_paths[i])
        sp = np.exp(lnsp_2d).astype(np.float32)

        # ── Geopotential (requires t + q + z_sfc simultaneously) ──
        print(f"      {ts_label}: computing geopotential at target pressure levels...")
        z_sfc = _read_sfc_var(z_sfc_paths[i])
        t_ml = _read_ml_var(ml_paths["t"][i]).astype(np.float32)
        q_ml = _read_ml_var(ml_paths["q"][i]).astype(np.float32)

        geo_at_levels = compute_geopotential_on_target_levels(
            t_ml, q_ml, z_sfc, lnsp_2d, LEVELS)

        del z_sfc  # free early

        for level in LEVELS:
            name = f"z_{level}"
            values_1d = ekr.interpolate(
                geo_at_levels[level], {"grid": (0.25, 0.25)}, {"grid": "N320"})
            field_arrays.setdefault(name, []).append(values_1d)
        del geo_at_levels

        # ── t and q: already loaded, interpolate before freeing ──
        # Compute p_full only for the levels we need (use mean sp for level finding,
        # but per-gridpoint for actual interpolation)
        for param, ml_data in [("t", t_ml), ("q", q_ml)]:
            print(f"      {ts_label}: interpolating {param}...")
            for level in LEVELS:
                target_pa = np.float32(level * 100.0)
                # Find bracketing model levels using mean sp
                p_full_mean = np.float32(
                    (A_HALF[:-1] + A_HALF[1:]) / 2 +
                    (B_HALF[:-1] + B_HALF[1:]) / 2 * sp.mean())
                below_idx = np.searchsorted(p_full_mean, target_pa)
                below_idx = min(max(below_idx, 1), 136)
                above_idx = below_idx - 1

                # Per-gridpoint pressure at bracketing levels
                p_below = (np.float32(A_HALF[below_idx]) + np.float32(B_HALF[below_idx]) * sp +
                           np.float32(A_HALF[below_idx + 1]) + np.float32(B_HALF[below_idx + 1]) * sp) / 2
                p_above = (np.float32(A_HALF[above_idx]) + np.float32(B_HALF[above_idx]) * sp +
                           np.float32(A_HALF[above_idx + 1]) + np.float32(B_HALF[above_idx + 1]) * sp) / 2

                # Log-pressure interpolation
                log_t = np.float32(np.log(target_pa))
                log_pb = np.log(p_below)
                log_pa = np.log(p_above)
                denom = log_pb - log_pa
                safe_denom = np.where(denom == 0, np.float32(1.0), denom)
                w = np.clip((log_t - log_pa) / safe_denom, 0.0, 1.0)

                values_2d = ml_data[above_idx] + w * (ml_data[below_idx] - ml_data[above_idx])
                values_1d = ekr.interpolate(
                    values_2d, {"grid": (0.25, 0.25)}, {"grid": "N320"})

                name = f"{param}_{level}"
                field_arrays.setdefault(name, []).append(values_1d)

        del t_ml, q_ml  # free ~1.8 GB

        # ── u and v: load one at a time ──
        for param in ["u", "v"]:
            print(f"      {ts_label}: interpolating {param}...")
            ml_data = _read_ml_var(ml_paths[param][i]).astype(np.float32)

            for level in LEVELS:
                target_pa = np.float32(level * 100.0)
                p_full_mean = np.float32(
                    (A_HALF[:-1] + A_HALF[1:]) / 2 +
                    (B_HALF[:-1] + B_HALF[1:]) / 2 * sp.mean())
                below_idx = np.searchsorted(p_full_mean, target_pa)
                below_idx = min(max(below_idx, 1), 136)
                above_idx = below_idx - 1

                p_below = (np.float32(A_HALF[below_idx]) + np.float32(B_HALF[below_idx]) * sp +
                           np.float32(A_HALF[below_idx + 1]) + np.float32(B_HALF[below_idx + 1]) * sp) / 2
                p_above = (np.float32(A_HALF[above_idx]) + np.float32(B_HALF[above_idx]) * sp +
                           np.float32(A_HALF[above_idx + 1]) + np.float32(B_HALF[above_idx + 1]) * sp) / 2

                log_t = np.float32(np.log(target_pa))
                log_pb = np.log(p_below)
                log_pa = np.log(p_above)
                denom = log_pb - log_pa
                safe_denom = np.where(denom == 0, np.float32(1.0), denom)
                w = np.clip((log_t - log_pa) / safe_denom, 0.0, 1.0)

                values_2d = ml_data[above_idx] + w * (ml_data[below_idx] - ml_data[above_idx])
                values_1d = ekr.interpolate(
                    values_2d, {"grid": (0.25, 0.25)}, {"grid": "N320"})

                name = f"{param}_{level}"
                field_arrays.setdefault(name, []).append(values_1d)

            del ml_data

        del sp

    # Stack timesteps into (2, N320)
    for name, arrs in field_arrays.items():
        fields[name] = np.stack(arrs)

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

    # ── Constant forcing fields (lsm, z, slor, sdor) from ECMWF Open Data ──
    print("\n  Fetching constant forcing fields...")
    const_fields = get_constant_fields()
    # Merge into shared PL fields (both are deterministic, shared across members)
    pl_fields.update(const_fields)
    print(f"  Total shared fields: {len(pl_fields)} ({len(pl_fields) - len(const_fields)} PL + {len(const_fields)} const)")

    print("\n" + "=" * 60)
    print("PHASE 2: Surface fields per EDA member + combine with shared fields")
    print("=" * 60)

    successful_members = []
    failed_members = []
    processing_times = []

    upload_enabled = UPLOAD_TO_GCS
    if upload_enabled and not os.path.exists(GCS_SERVICE_ACCOUNT_KEY):
        print(f"GCS key not found: {GCS_SERVICE_ACCOUNT_KEY}, disabling upload")
        upload_enabled = False

    datestr = date.strftime("%Y%m%d")

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
            expected_count = len(PARAM_SFC_EDA) + len(PARAM_PL) * len(LEVELS) + len(PARAM_SFC_CONST)
            actual_count = len(fields)
            print(f"  Fields: {actual_count}/{expected_count} "
                  f"({len(PARAM_SFC_EDA)} sfc + {len(PARAM_PL)*len(LEVELS)} pl + {len(PARAM_SFC_CONST)} const)")

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
                    gcs_blob_name = f"era5t/{datestr}/input_state_member_{member:03d}.pkl"
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
    n_total = len(PARAM_SFC_EDA) + len(PARAM_PL) * len(LEVELS) + len(PARAM_SFC_CONST)
    print(f"Fields per member: {len(PARAM_SFC_EDA)} sfc + {len(PARAM_PL)*len(LEVELS)} pl + "
          f"{len(PARAM_SFC_CONST)} const = {n_total} total")
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
