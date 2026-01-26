#!/usr/bin/env python3
"""
GRIB to NetCDF Processor — CLI version with configurable date, members, and FP16 support
========================================================================================

Usage:
    # FP32 mode (default): reads from forecasts/, writes to 1p5deg_nc/
    python aifs_n320_grib_1p5defg_nc_cli.py --date 20251127_0000 --members 1-50

    # FP16 mode: reads from fp16_forecasts/, writes to fp16_1p5deg_nc/
    python aifs_n320_grib_1p5defg_nc_cli.py --date 20251127_0000 --members 1-50 --fp16

GCS Path Structure:
    FP32: gs://bucket/{date}/forecasts/     -> gs://bucket/{date}/1p5deg_nc/
    FP16: gs://bucket/{date}/fp16_forecasts/ -> gs://bucket/{date}/fp16_1p5deg_nc/
"""

import os
import gc
import time
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import List, Optional

# -----------------------------------------------------------------------------
# IMPORTANT: Configure paths and earthkit settings BEFORE importing earthkit
# This ensures the cache settings are applied before the singleton is created
# -----------------------------------------------------------------------------
DEF_BASE = "/scratch/notebook"
BASE_DIR = Path(os.environ.get("EARTHKIT_WORKDIR", DEF_BASE))
TMP_DIR = BASE_DIR / "tmp"
EK_CACHE_DIR = BASE_DIR / ".cache/earthkit-data"
EK_TMP_DIR = BASE_DIR / "earthkit-tmp"
HOME_CACHE_DIR = Path.home() / ".cache"
EK_REGRID_CACHE = HOME_CACHE_DIR / "earthkit-regrid"

# Ensure directories exist
for p in [TMP_DIR, EK_CACHE_DIR, EK_TMP_DIR, EK_REGRID_CACHE]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Force env vars BEFORE importing earthkit
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["XDG_CACHE_HOME"] = str(BASE_DIR / ".cache")
os.environ["ECCODES_TMPDIR"] = str(EK_TMP_DIR)
# Disable tqdm progress bars which can hang in some terminal contexts
os.environ["TQDM_DISABLE"] = "1"

# Configure earthkit-regrid cache settings BEFORE importing earthkit.regrid
# Use "user" cache policy with persistent storage - the matrix files will be cached
# and reused across all members. We fixed cleanup_earthkit_dirs() to NOT delete
# the regrid cache during processing, which was causing SQLite corruption.
try:
    from earthkit.regrid.utils import caching as regrid_caching
    # Use persistent user cache (NOT "off") - matrix files cached on disk
    regrid_caching.SETTINGS["cache-policy"] = "user"
    regrid_caching.SETTINGS["user-cache-directory"] = str(EK_REGRID_CACHE)
    # Increase download timeout for slow connections to ECMWF servers
    regrid_caching.SETTINGS["url-download-timeout"] = 300  # 5 minutes instead of 30 seconds
    # Enable in-memory caching of regrid matrices for faster repeated access
    regrid_caching.SETTINGS["matrix-memory-cache-policy"] = "lru"
    regrid_caching.SETTINGS["maximum-matrix-memory-cache-size"] = 2 * 1024 * 1024 * 1024  # 2GB
except Exception:
    pass

# Now import the rest of the dependencies
from google.cloud import storage
import xarray as xr
import numpy as np

import earthkit.data as ekd
import earthkit.regrid as ekr

# Earthkit-data settings (cache directory) - can be set after import
try:
    from earthkit.data import settings as ek_settings
    ek_settings.set("cache.directory", str(EK_CACHE_DIR))
except Exception:
    pass


def get_disk_free_gb() -> float:
    """Return free disk space in GB for the scratch partition."""
    import subprocess
    try:
        result = subprocess.run(['df', '-BG', '/scratch'], capture_output=True, text=True)
        # Parse output: Filesystem 1G-blocks Used Available Use% Mounted
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            available = int(parts[3].rstrip('G'))
            return available
    except Exception:
        pass
    return -1


def diagnose_disk_usage() -> None:
    """Print top directories consuming disk space for debugging."""
    import subprocess
    print("    📊 Disk usage diagnosis:")
    dirs_to_check = [
        "/scratch/notebook/tmp",
        "/scratch/notebook/.cache",
        "/scratch/notebook/earthkit-tmp",
        str(Path.home() / ".cache"),
        "/tmp",
        "/var/tmp",
    ]
    for d in dirs_to_check:
        try:
            result = subprocess.run(['du', '-sh', d], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                size = result.stdout.strip().split()[0]
                print(f"       {d}: {size}")
        except Exception:
            pass


def parse_member_range(member_str: str) -> List[int]:
    """Parse member range string like '1-50' or '1,2,3' into list of integers."""
    members = []
    if '-' in member_str:
        start, end = map(int, member_str.split('-'))
        members = list(range(start, end + 1))
    elif ',' in member_str:
        members = [int(m.strip()) for m in member_str.split(',')]
    else:
        members = [int(member_str)]
    return members


class GRIBToNetCDFProcessor:
    def __init__(self, date_str: str, members: List[int], fp16: bool = False, skip_upload: bool = False):
        """
        Initialize processor with configurable date, members, and precision mode.

        Args:
            date_str: Date string in format YYYYMMDD_0000
            members: List of ensemble member numbers (1-indexed)
            fp16: If True, use FP16 paths (fp16_forecasts/, fp16_1p5deg_nc/)
            skip_upload: If True, skip uploading NetCDF files to GCS (for testing)
        """
        self.skip_upload = skip_upload
        # GCS Configuration
        self.gcs_bucket = "aifs-aiquest-us-20251127"
        self.service_account_key = "coiled-data.json"

        # Parse date string
        if '_' in date_str:
            self.forecast_date = date_str.split('_')[0]
            self.forecast_time = date_str.split('_')[1]
        else:
            self.forecast_date = date_str
            self.forecast_time = "0000"

        self.date_prefix = f"{self.forecast_date}_{self.forecast_time}"

        # Set paths based on FP16 flag
        self.fp16 = fp16
        if fp16:
            self.gcs_input_prefix = f"{self.date_prefix}/fp16_forecasts/"
            self.gcs_output_prefix = f"{self.date_prefix}/fp16_1p5deg_nc/"
            self.mode_label = "FP16"
        else:
            self.gcs_input_prefix = f"{self.date_prefix}/forecasts/"
            self.gcs_output_prefix = f"{self.date_prefix}/1p5deg_nc/"
            self.mode_label = "FP32"

        # Ensemble members
        self.members = members

        # Time ranges for 792-hour forecast
        self.time_ranges = [
            ("432", "504"),  # Days 18-21
            ("504", "576"),  # Days 21-24
            ("576", "648"),  # Days 24-27
            ("648", "720"),  # Days 27-30
            ("720", "792"),  # Days 30-33
        ]

        # Variable mapping for NetCDF conversion
        self.var_mapping = {
            "mslp": "msl",  # Mean sea level pressure
            "pr": "tp",      # Total precipitation
            "tas": "2t",     # 2-meter temperature
        }

        # GCS client
        self.client: Optional[storage.Client] = None
        self.bucket = None

        # Temporary directory for processing
        self.temp_dir: Optional[str] = None

    def initialize_gcs(self) -> bool:
        try:
            print("🔗 Initializing GCS connection...")
            self.client = storage.Client.from_service_account_json(
                self.service_account_key
            )
            self.bucket = self.client.bucket(self.gcs_bucket)
            print(f"✅ Connected to GCS bucket: {self.gcs_bucket}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize GCS: {e}")
            return False

    def create_temp_directory(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="grib_nc_processor_", dir=str(TMP_DIR))
        print(f"📁 Created temporary directory: {self.temp_dir}")

    def cleanup_temp_directory(self) -> None:
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print("🧹 Cleaned up temporary directory")
            except Exception as e:
                print(f"⚠️  Could not remove temp directory {self.temp_dir}: {e}")
        self.temp_dir = None

    def download_grib_file(self, member: int, time_range) -> Optional[str]:
        start_hour, end_hour = time_range
        filename = (
            f"aifs_ens_forecast_{self.forecast_date}_{self.forecast_time}_"
            f"member{member:03d}_h{start_hour}-{end_hour}.grib"
        )
        blob_name = f"{self.gcs_input_prefix}{filename}"
        local_path = os.path.join(self.temp_dir, filename)
        print(f"    Looking for: {blob_name}")
        try:
            blob = self.bucket.get_blob(blob_name)
            if blob is None:
                print(f"    ⚠️  File not found: {blob_name}")
                return None

            print(f"    ⬇️  Downloading: {blob_name}")
            blob.download_to_filename(local_path)
            print(f"    ✅ Downloaded: {os.path.getsize(local_path)/(1024*1024):.1f} MB")
            return local_path
        except Exception as e:
            print(f"    ❌ Download failed: {e}")
            return None

    @staticmethod
    def _safe_close(obj) -> None:
        """Close Earthkit FieldList-like objects if they expose .close()."""
        try:
            close = getattr(obj, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    def process_single_grib(self, grib_file: str) -> Optional[xr.Dataset]:
        """Process a single GRIB file and return extracted dataset (loaded in memory)."""
        fl = fl_ll_1p5 = None
        ds = extracted_ds = result = None
        try:
            # Open GRIB (Earthkit FieldList)
            fl = ekd.from_source("file", grib_file)

            # Regrid from N320 to 1.5° regular lon/lat
            fl_ll_1p5 = ekr.interpolate(
                fl, in_grid={"grid": "N320"}, out_grid={"grid": [1.5, 1.5]}
            )

            # Convert to xarray and detach from files
            ds = fl_ll_1p5.to_xarray()
            ds.load()

            available_vars = list(ds.data_vars)
            target_vars = [
                alt_name for _, alt_name in self.var_mapping.items() if alt_name in available_vars
            ]

            if target_vars:
                extracted_ds = ds[target_vars]
                extracted_ds.load()
                # Make a copy to keep in memory after closing
                result = extracted_ds.copy(deep=True)
                print(f"      ✓ Extracted: {target_vars}")
            else:
                print("      ⚠️  No target variables found")

        except Exception as e:
            print(f"      ❌ Error processing GRIB: {e}")
        finally:
            self._safe_close(fl_ll_1p5)
            self._safe_close(fl)
            for obj in [extracted_ds, ds]:
                try:
                    if hasattr(obj, "close"):
                        obj.close()
                except Exception:
                    pass
            del extracted_ds, ds, fl_ll_1p5, fl
            gc.collect()
        return result

    def process_grib_to_netcdf(self, member: int) -> Optional[str]:
        """Download, process, and immediately delete each GRIB file to minimize disk usage."""
        print("  🔄 Downloading and converting GRIB files one at a time...")

        member_datasets: List[xr.Dataset] = []
        try:
            for i, time_range in enumerate(self.time_ranges):
                # Download single GRIB file
                grib_file = self.download_grib_file(member, time_range)
                if not grib_file or not os.path.exists(grib_file):
                    print(f"      ⚠️  Skipping time range {time_range}")
                    continue

                print(f"    Processing ({i+1}/{len(self.time_ranges)}): {os.path.basename(grib_file)}")

                # Process and extract data to memory
                extracted_ds = self.process_single_grib(grib_file)

                # Immediately delete the GRIB file to free disk space
                try:
                    os.remove(grib_file)
                    print(f"      🗑️  Removed GRIB file (freed ~1.4GB)")
                except Exception as e:
                    print(f"      ⚠️  Could not remove {grib_file}: {e}")

                # Clean earthkit temp dirs after each file (preserve regrid cache for next file)
                self.cleanup_earthkit_dirs(verbose=False, full_cleanup=False)
                gc.collect()

                # Report disk space after each file
                free_gb = get_disk_free_gb()
                print(f"      💾 Disk free: {free_gb} GB")
                if free_gb < 8:  # Diagnose if running low
                    diagnose_disk_usage()

                if extracted_ds is not None:
                    member_datasets.append(extracted_ds)

            if not member_datasets:
                print(f"    ⚠️  No valid datasets for member {member:03d}")
                return None

            # Concatenate along time
            member_combined = xr.concat(member_datasets, dim="time")
            for ds_part in member_datasets:
                try:
                    if hasattr(ds_part, "close"):
                        ds_part.close()
                except Exception:
                    pass
            del member_datasets
            gc.collect()

            # Add member dimension
            member_combined = member_combined.expand_dims("member").assign_coords(
                member=[f"{member:03d}"]
            )

            # Metadata
            member_combined.attrs.update(
                {
                    "title": f"AIFS Ensemble Forecast Data ({self.mode_label})",
                    "description": f"Regridded to 1.5 degree resolution, member {member:03d}",
                    "source": f"ECMWF AIFS ensemble forecast ({self.mode_label})",
                    "grid_resolution": "1.5 degrees",
                    "forecast_date": f"{self.forecast_date} {self.forecast_time}:00",
                    "member": f"member{member:03d}",
                    "precision": self.mode_label,
                    "variables": ", ".join(self.var_mapping.keys()),
                    "processing_date": str(np.datetime64("now")),
                }
            )

            # Clean attributes
            member_combined = self.clean_dataset_attrs(member_combined)

            # Save to NetCDF
            nc_filename = f"aifs_ensemble_forecast_1p5deg_member{member:03d}.nc"
            nc_path = os.path.join(self.temp_dir, nc_filename)
            print(f"    💾 Saving NetCDF: {nc_filename}")
            member_combined.to_netcdf(nc_path, engine="netcdf4")

            try:
                if hasattr(member_combined, "close"):
                    member_combined.close()
            except Exception:
                pass
            del member_combined
            gc.collect()

            size_mb = os.path.getsize(nc_path) / (1024 * 1024)
            print(f"    ✅ NetCDF created: {size_mb:.1f} MB")
            return nc_path

        except Exception as e:
            print(f"    ❌ NetCDF conversion failed: {e}")
            return None

    def clean_dataset_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        def clean_attrs(obj):
            if hasattr(obj, "attrs"):
                attrs_to_remove = []
                for key, value in obj.attrs.items():
                    if key.startswith("_earthkit") or isinstance(value, dict):
                        attrs_to_remove.append(key)
                for key in attrs_to_remove:
                    del obj.attrs[key]
            return obj

        ds = clean_attrs(ds)
        for var_name in list(ds.data_vars):
            ds[var_name] = clean_attrs(ds[var_name])
        for coord_name in list(ds.coords):
            ds[coord_name] = clean_attrs(ds[coord_name])
        return ds

    def upload_netcdf_to_gcs(self, nc_path: str, member: int) -> bool:
        try:
            nc_filename = os.path.basename(nc_path)
            blob_name = f"{self.gcs_output_prefix}{nc_filename}"

            print(f"    ⬆️  Uploading: {nc_filename}")
            start_time = time.time()

            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(nc_path)

            upload_time = time.time() - start_time
            file_size = os.path.getsize(nc_path) / (1024 * 1024)

            print(f"    ✅ Uploaded: {file_size:.1f} MB in {upload_time:.1f}s")
            print(f"    📍 GCS path: gs://{self.gcs_bucket}/{blob_name}")
            return True
        except Exception as e:
            print(f"    ❌ Upload failed: {e}")
            return False

    def cleanup_local_files(self, files_to_remove: List[Optional[str]]) -> None:
        gc.collect()
        for file_path in files_to_remove:
            if not file_path:
                continue
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"    🗑️  Removed: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"    ⚠️  Could not remove {file_path}: {e}")

    def cleanup_earthkit_dirs(self, verbose: bool = True, full_cleanup: bool = True) -> None:
        """Clean up earthkit cache and temp directories to free disk space.

        Args:
            verbose: Print cleanup messages
            full_cleanup: If True, clean more aggressively. Note: we NEVER delete the
                          earthkit-regrid cache directory because it uses a SQLite
                          singleton that becomes corrupted if the directory is deleted
                          while the process is running.
        """
        # Clean main earthkit directories (always clean these)
        for dir_path, label in [
            (EK_CACHE_DIR, "earthkit cache"),
            (EK_TMP_DIR, "earthkit tmp"),
        ]:
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True, exist_ok=True)
                    if verbose:
                        print(f"    🧹 Cleared {label} directory")
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  Could not clear {label}: {e}")

        # IMPORTANT: Do NOT delete EK_REGRID_CACHE here!
        # The earthkit-regrid Cache is a singleton with a persistent SQLite connection.
        # Deleting the directory corrupts the singleton state, causing "readonly database"
        # errors on subsequent members. The regrid matrices are needed for all members
        # and use the same grid transformation, so caching them is beneficial.
        #
        # If disk space is critical, clean regrid cache only between separate script runs,
        # not between members within a single run.

        # Clean any earthkit/eccodes files in home .cache (but NOT earthkit-regrid)
        try:
            if HOME_CACHE_DIR.exists():
                for item in HOME_CACHE_DIR.iterdir():
                    # NEVER delete earthkit-regrid - it will corrupt the singleton
                    if item.name == "earthkit-regrid":
                        continue
                    if item.name.startswith(("earthkit", "eccodes")):
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
        except Exception:
            pass

        # Also clean system /tmp which ecCodes/earthkit may use
        try:
            for item in Path("/tmp").iterdir():
                if item.name.startswith(("eccodes", "earthkit", "grib", "tmp")):
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        except Exception:
            pass

        # Clean the TMP_DIR contents (our temp directory for downloads)
        if full_cleanup:
            try:
                if TMP_DIR.exists():
                    for item in TMP_DIR.iterdir():
                        try:
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                        except Exception:
                            pass
                    if verbose:
                        print(f"    🧹 Cleared TMP_DIR contents")
            except Exception:
                pass

    def process_member(self, member: int) -> bool:
        print("\n" + "=" * 60)
        print(f"Processing Member {member:03d} ({self.mode_label})")
        print("=" * 60)

        # Report disk space at start
        free_gb = get_disk_free_gb()
        print(f"💾 Disk space available: {free_gb} GB")

        # Minimum disk space check - each GRIB file is ~1.4GB, need buffer for processing
        MIN_DISK_GB = 3
        if free_gb < MIN_DISK_GB:
            print(f"❌ Insufficient disk space ({free_gb} GB < {MIN_DISK_GB} GB minimum)")
            print("   Try running members in smaller batches or clearing disk space.")
            diagnose_disk_usage()
            return False

        success = False
        nc_file: Optional[str] = None

        try:
            # Step 1 & 2: Download and convert GRIB files one at a time
            print(f"📥 Processing GRIB files for member {member:03d}")
            nc_file = self.process_grib_to_netcdf(member)
            if not nc_file:
                print(f"  ❌ NetCDF conversion failed for member {member:03d}")
                return False

            # Step 3: Upload NetCDF to GCS (skip if --no-upload flag is set)
            if self.skip_upload:
                print("☁️  Step 3: Skipping upload (--no-upload flag set)")
                file_size = os.path.getsize(nc_file) / (1024 * 1024)
                print(f"    📄 NetCDF file ready: {os.path.basename(nc_file)} ({file_size:.1f} MB)")
            else:
                print("☁️  Step 3: Uploading NetCDF to GCS")
                if not self.upload_netcdf_to_gcs(nc_file, member):
                    print(f"  ❌ Upload failed for member {member:03d}")
                    return False

            success = True
            print(f"  ✅ Member {member:03d} processed successfully!")

        except Exception as e:
            print(f"  ❌ Error processing member {member:03d}: {e}")
            success = False
        finally:
            # Step 4: Cleanup NetCDF file and earthkit directories
            print("🧹 Step 4: Final cleanup")
            if nc_file:
                self.cleanup_local_files([nc_file])
            self.cleanup_earthkit_dirs()
            gc.collect()

            # Report disk space after cleanup
            free_gb = get_disk_free_gb()
            print(f"💾 Disk space after cleanup: {free_gb} GB")

        return success

    def run(self) -> int:
        print("=" * 70)
        print(f"GRIB to NetCDF Processor ({self.mode_label} Mode)")
        print("=" * 70)
        print(f"Bucket: {self.gcs_bucket}")
        print(f"Input path: {self.gcs_input_prefix}")
        print(f"Output path: {self.gcs_output_prefix}")
        print(f"Members: {min(self.members)}-{max(self.members)} ({len(self.members)} total)")
        print(f"Time ranges: {len(self.time_ranges)} periods")
        print(f"Forecast: {self.forecast_date} {self.forecast_time}")
        print(f"Precision: {self.mode_label}")
        if self.skip_upload:
            print(f"Upload: DISABLED (--no-upload flag)")

        # Initial disk space and cleanup check
        free_gb = get_disk_free_gb()
        print(f"💾 Initial disk space: {free_gb} GB")
        if free_gb < 5:
            print("⚠️  Low disk space, performing initial cleanup...")
            # Clear stale caches from previous runs (but NOT regrid cache - it has valuable matrix files)
            for cleanup_dir in [EK_CACHE_DIR, EK_TMP_DIR, TMP_DIR]:
                try:
                    if cleanup_dir.exists():
                        shutil.rmtree(cleanup_dir)
                        cleanup_dir.mkdir(parents=True, exist_ok=True)
                        print(f"    🧹 Cleared {cleanup_dir}")
                except Exception:
                    pass
            # NOTE: Do NOT delete EK_REGRID_CACHE here - it contains cached regrid matrices
            # that take a long time to download from ECMWF servers
            free_gb = get_disk_free_gb()
            print(f"💾 Disk space after cleanup: {free_gb} GB")

        if not self.initialize_gcs():
            return 1

        self.create_temp_directory()

        successful_members: List[int] = []
        failed_members: List[int] = []
        start_time = time.time()

        try:
            for i, member in enumerate(self.members):
                print(f"\n🔄 Processing member {i + 1}/{len(self.members)}")
                member_start_time = time.time()
                ok = self.process_member(member)
                took = time.time() - member_start_time
                (successful_members if ok else failed_members).append(member)

                # Progress estimate
                elapsed_total = time.time() - start_time
                avg = elapsed_total / (i + 1)
                remaining = avg * (len(self.members) - i - 1)
                print(f"  ⏱️  Member time: {took/60:.1f} min")
                print(f"  ⏱️  Estimated remaining: {remaining/60:.1f} min")

                gc.collect()

        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
        finally:
            self.cleanup_temp_directory()

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"PROCESSING SUMMARY ({self.mode_label})")
        print("=" * 70)
        print(f"✅ Successful: {len(successful_members)}/{len(self.members)} members")
        print(f"❌ Failed: {len(failed_members)} members")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")

        if failed_members:
            print(f"\nFailed members: {failed_members}")
        if successful_members:
            print(f"\nNetCDF files uploaded to: gs://{self.gcs_bucket}/{self.gcs_output_prefix}")

        return 0 if not failed_members else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GRIB to NetCDF Processor with CLI arguments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GCS Path Structure:
    FP32 (default):
        Input:  gs://bucket/{date}/forecasts/
        Output: gs://bucket/{date}/1p5deg_nc/

    FP16 (--fp16 flag):
        Input:  gs://bucket/{date}/fp16_forecasts/
        Output: gs://bucket/{date}/fp16_1p5deg_nc/

Examples:
    # Process members 1-50 for FP32 forecasts
    python aifs_n320_grib_1p5defg_nc_cli.py --date 20251127_0000 --members 1-50

    # Process members 1-50 for FP16 forecasts
    python aifs_n320_grib_1p5defg_nc_cli.py --date 20251127_0000 --members 1-50 --fp16

    # Process specific members
    python aifs_n320_grib_1p5defg_nc_cli.py --date 20251127_0000 --members 1,5,10,25

    # Test processing without uploading to GCS
    python aifs_n320_grib_1p5defg_nc_cli.py --date 20251127_0000 --members 1-3 --no-upload
        """
    )

    parser.add_argument('--date', required=True,
                       help='Date string (YYYYMMDD_0000 or YYYYMMDD)')
    parser.add_argument('--members', default='1-50',
                       help='Member range (e.g., 1-50, 1,2,3)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 paths (fp16_forecasts/ -> fp16_1p5deg_nc/)')
    parser.add_argument('--no-upload', action='store_true',
                       help='Skip uploading NetCDF files to GCS (for testing)')
    parser.add_argument('--bucket', default='aifs-aiquest-us-20251127',
                       help='GCS bucket name')
    parser.add_argument('--service-account', default='coiled-data.json',
                       help='Path to GCS service account key')

    args = parser.parse_args()

    # Parse members
    try:
        members = parse_member_range(args.members)
        print(f"Processing {len(members)} members: {members[0]}-{members[-1]}")
    except ValueError as e:
        print(f"ERROR: Invalid member range: {e}")
        return 1

    # Create and run processor
    processor = GRIBToNetCDFProcessor(
        date_str=args.date,
        members=members,
        fp16=args.fp16,
        skip_upload=args.no_upload
    )
    return processor.run()


if __name__ == "__main__":
    exit(main())
