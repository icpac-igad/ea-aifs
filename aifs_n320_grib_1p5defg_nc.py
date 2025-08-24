#!/usr/bin/env python3
"""
GRIB to NetCDF Processor — leak-safe revision
=============================================

Key changes to prevent "deleted but still open" temp-file leaks that were
filling the filesystem (df -h ~38G used while du shows small):

1) **No debugger stops** in the hot path (removed ipdb.set_trace()).
2) **Explicitly close Earthkit FieldList objects** (both the original and the
   regridded ones) using try/finally and `.close()` when present.
3) **Ensure Xarray datasets don't hold file handles** by calling `.load()`
   to detach from any file-backed objects, then letting GC collect them
   (and calling `.close()` if available).
4) **Order of cleanup**: close handles first, then unlink files/dirs.
5) **Configurable temp/cache locations** so Earthkit/ecCodes won't spill into
   ephemeral `/tmp`; defaults target `/scratch/notebook` if available.
6) **Extra GC passes** after each member to hasten FD release.

If still using Dask elsewhere, start workers with explicit spill dirs, or set:
  DASK_TEMPORARY_DIRECTORY=/scratch/notebook/dask-tmp
  DASK_WORKER_TEMPORARY_DIRECTORY=/scratch/notebook/dask-tmp

"""

import os
import gc
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

from google.cloud import storage
import xarray as xr
import numpy as np

import earthkit.data as ekd
import earthkit.regrid as ekr


# -----------------------------------------------------------------------------
# Helper: choose safe local dirs for tmp/cache (avoids root overlay saturation)
# -----------------------------------------------------------------------------
DEF_BASE = "/scratch/notebook"
BASE_DIR = Path(os.environ.get("EARTHKIT_WORKDIR", DEF_BASE))
TMP_DIR = BASE_DIR / "tmp"
EK_CACHE_DIR = BASE_DIR / ".cache/earthkit-data"
EK_TMP_DIR = BASE_DIR / "earthkit-tmp"

# Ensure directories exist; if not, fallback to system defaults but still try
# to keep under /scratch/notebook when possible.
for p in [TMP_DIR, EK_CACHE_DIR, EK_TMP_DIR]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Steer Python and Earthkit/ecCodes temp/cache paths via env vars early
os.environ.setdefault("TMPDIR", str(TMP_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))
os.environ.setdefault("ECCODES_TMPDIR", str(EK_TMP_DIR))

# Earthkit settings (cache directory). Some versions allow tmp settings too.
try:
    from earthkit.data import settings as ek_settings
    ek_settings.set("cache.directory", str(EK_CACHE_DIR))
    # If your earthkit version supports it, uncomment the next line:
    # ek_settings.set("tmp.directory", str(EK_TMP_DIR))
except Exception:
    pass


class GRIBToNetCDFProcessor:
    def __init__(self):
        # GCS Configuration
        self.gcs_bucket = "ea_aifs_w1"
        self.gcs_input_prefix = "forecasts/20240821_0000/"
        self.gcs_output_prefix = "20240821_0000/1p5deg_nc/"
        self.service_account_key = "coiled-data-e4drr_202505.json"

        # Forecast Configuration
        self.forecast_date = "20250822"
        self.forecast_time = "1200"
        #self.forecast_time = "0000" for ensemble 1-28

        self.members = list(range(36, 51))  # Members 1-50

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
        # Create temp dir under our controlled base
        self.temp_dir = tempfile.mkdtemp(prefix="grib_nc_processor_", dir=str(TMP_DIR))
        print(f"📁 Created temporary directory: {self.temp_dir}")

    def cleanup_temp_directory(self) -> None:
        # Remove only after everything is closed
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
        print(blob_name)
        try:
            blob = self.bucket.get_blob(blob_name)  # <-- robust existence check
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
            
    @staticmethod
    def truncate_deleted_open_files(pids=None, process_name_filters=None, dry_run=False, verbose=True):
        """
        Find and (optionally) truncate all *deleted-but-still-open* regular files
        held by running processes, freeing disk space without killing the process.
    
        Parameters
        ----------
        pids : None | int | list[int]
            If None, scan all PIDs under /proc. Otherwise, only scan the given PID
            or list of PIDs.
        process_name_filters : None | list[str]
            If provided, only act on processes whose /proc/<pid>/{comm,cmdline}
            contains any of these substrings (case-insensitive), e.g. ["python","dask","earthkit"].
        dry_run : bool
            If True, only print what would be truncated; do not modify anything.
        verbose : bool
            If True, print findings and actions.
    
        Returns
        -------
        dict
            {'truncated_fds': int, 'approx_bytes_freed': int}  (bytes are approximate)
        """
        import os
        import stat as pystat
    
        def _match_proc(pid: int) -> bool:
            if not process_name_filters:
                return True
            try:
                with open(f"/proc/{pid}/comm", "rb") as f:
                    comm = f.read().decode(errors="ignore").strip()
            except Exception:
                comm = ""
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().decode(errors="ignore").replace("\x00", " ").strip()
            except Exception:
                cmd = ""
            text = (comm + " " + cmd).lower()
            return any(s.lower() in text for s in process_name_filters)
    
        # Build PID list
        if pids is None:
            pid_list = []
            for name in os.listdir("/proc"):
                if name.isdigit():
                    pid_list.append(int(name))
        elif isinstance(pids, int):
            pid_list = [pids]
        else:
            pid_list = list(map(int, pids))
    
        truncated = 0
        freed = 0
    
        for pid in pid_list:
            if not _match_proc(pid):
                continue
            fd_dir = f"/proc/{pid}/fd"
            try:
                fds = os.listdir(fd_dir)
            except Exception:
                continue
    
            for fd in fds:
                fd_path = f"{fd_dir}/{fd}"
                try:
                    target = os.readlink(fd_path)
                except Exception:
                    continue
    
                # Only act on deleted regular files
                if not target.endswith(" (deleted)"):
                    continue
                try:
                    st = os.stat(fd_path)
                except Exception:
                    continue
                if not pystat.S_ISREG(st.st_mode):
                    continue
    
                if verbose:
                    print(f"PID {pid} FD {fd} -> {target} size={st.st_size} bytes")
    
                if dry_run:
                    continue
    
                # Truncate via the proc FD handle
                try:
                    fdw = os.open(fd_path, os.O_WRONLY)
                    try:
                        os.ftruncate(fdw, 0)
                    finally:
                        os.close(fdw)
                    truncated += 1
                    # st_size can be 0 for some files; treat as approximate
                    freed += getattr(st, "st_size", 0) or 0
                    if verbose:
                        print(f"  truncated /proc/{pid}/fd/{fd}")
                except Exception as e:
                    if verbose:
                        print(f"  skip: {e}")
    
        return {"truncated_fds": truncated, "approx_bytes_freed": freed}


    def process_grib_to_netcdf(self, member: int, grib_files: List[str]) -> Optional[str]:
        print("  🔄 Converting GRIB to NetCDF...")

        member_datasets: List[xr.Dataset] = []
        try:
            for grib_file in grib_files:
                if not grib_file or not os.path.exists(grib_file):
                    continue

                print(f"    Processing: {os.path.basename(grib_file)}")
                fl = fl_ll_1p5 = None
                ds = extracted_ds = None
                try:
                    # Open GRIB (Earthkit FieldList)
                    fl = ekd.from_source("file", grib_file)

                    # Regrid from N320 to 1.5° regular lon/lat
                    fl_ll_1p5 = ekr.interpolate(
                        fl, in_grid={"grid": "N320"}, out_grid={"grid": [1.5, 1.5]}
                    )

                    # Convert to xarray and detach from files
                    ds = fl_ll_1p5.to_xarray()
                    # Load into memory so no file handles linger
                    ds.load()

                    available_vars = list(ds.data_vars)
                    target_vars = [
                        alt_name for _, alt_name in self.var_mapping.items() if alt_name in available_vars
                    ]

                    if target_vars:
                        extracted_ds = ds[target_vars]
                        # Also force compute in case of dask-backed
                        extracted_ds.load()
                        member_datasets.append(extracted_ds)
                        print(f"      ✓ Extracted: {target_vars}")
                    else:
                        print("      ⚠️  No target variables found")

                except Exception as e:
                    print(f"      ❌ Error processing GRIB: {e}")
                finally:
                    # Try closing Earthkit handles first
                    self._safe_close(fl_ll_1p5)
                    self._safe_close(fl)
                    # Xarray datasets — no .close() on in-memory, but try anyway
                    for obj in [extracted_ds, ds]:
                        try:
                            if hasattr(obj, "close"):
                                obj.close()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    # Encourage early GC
                    del extracted_ds, ds, fl_ll_1p5, fl
                    gc.collect()

            if not member_datasets:
                print(f"    ⚠️  No valid datasets for member {member:03d}")
                return None

            # Concatenate along time
            member_combined = xr.concat(member_datasets, dim="time")
            # Free partials asap
            for ds_part in member_datasets:
                try:
                    if hasattr(ds_part, "close"):
                        ds_part.close()  # type: ignore[attr-defined]
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
                    "title": "AIFS Ensemble Forecast Data",
                    "description": f"Regridded to 1.5 degree resolution, member {member:03d}",
                    "source": "ECMWF AIFS ensemble forecast",
                    "grid_resolution": "1.5 degrees",
                    "forecast_date": f"{self.forecast_date} {self.forecast_time}:00",
                    "member": f"member{member:03d}",
                    "variables": ", ".join(self.var_mapping.keys()),
                    "processing_date": str(np.datetime64("now")),
                }
            )

            # Clean attributes to avoid NetCDF serialization issues
            member_combined = self.clean_dataset_attrs(member_combined)

            # Save to NetCDF
            nc_filename = f"aifs_ensemble_forecast_1p5deg_member{member:03d}.nc"
            nc_path = os.path.join(self.temp_dir, nc_filename)
            print(f"    💾 Saving NetCDF: {nc_filename}")
            member_combined.to_netcdf(nc_path, engine="netcdf4")

            # After writing, close and free
            try:
                if hasattr(member_combined, "close"):
                    member_combined.close()  # type: ignore[attr-defined]
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
            print(f"    📍 GCS path: gs://{self.gcs_bucket}/{self.gcs_output_prefix}{nc_filename}")
            return True
        except Exception as e:
            print(f"    ❌ Upload failed: {e}")
            return False

    def cleanup_local_files(self, files_to_remove: List[Optional[str]]) -> None:
        # Ensure handles are closed before attempting unlink
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

    def process_member(self, member: int) -> bool:
        print("\n" + "=" * 60)
        print(f"Processing Member {member:03d}")
        print("=" * 60)

        success = False
        grib_files: List[Optional[str]] = []
        nc_file: Optional[str] = None

        try:
            # Step 1: Download GRIB files
            print(f"📥 Step 1: Downloading GRIB files for member {member:03d}")
            for time_range in self.time_ranges:
                grib_files.append(self.download_grib_file(member, time_range))

            valid_grib_files = [f for f in grib_files if f]
            if not valid_grib_files:
                print(f"  ❌ No valid GRIB files downloaded for member {member:03d}")
                return False

            print(f"  ✅ Downloaded {len(valid_grib_files)}/{len(self.time_ranges)} GRIB files")

            # Step 2: Convert to NetCDF
            print("🔄 Step 2: Converting to NetCDF")
            nc_file = self.process_grib_to_netcdf(member, valid_grib_files)
            if not nc_file:
                print(f"  ❌ NetCDF conversion failed for member {member:03d}")
                return False

            # Step 3: Upload NetCDF to GCS
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
            # Step 4: Cleanup local files AFTER all objects are closed
            print("🧹 Step 4: Cleaning up local files")
            files_to_cleanup: List[Optional[str]] = grib_files + ([nc_file] if nc_file else [])
            self.cleanup_local_files(files_to_cleanup)
            gc.collect()

        return success

    def run(self) -> int:
        print("GRIB to NetCDF Processor (leak-safe)")
        print("=" * 60)
        print(f"Bucket: {self.gcs_bucket}")
        print(f"Input path: {self.gcs_input_prefix}")
        print(f"Output path: {self.gcs_output_prefix}")
        print(f"Members: {min(self.members)}-{max(self.members)}")
        print(f"Time ranges: {len(self.time_ranges)} periods")
        print(f"Forecast: {self.forecast_date} {self.forecast_time}")

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

                # Extra GC after each member
                gc.collect()
                self.truncate_deleted_open_files(process_name_filters=["python","earthkit","dask"])

        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
        finally:
            self.cleanup_temp_directory()

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"✅ Successful: {len(successful_members)}/{len(self.members)} members")
        print(f"❌ Failed: {len(failed_members)} members")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")

        if failed_members:
            print(f"\nFailed members: {failed_members}")
        if successful_members:
            print(f"\nNetCDF files uploaded to: gs://{self.gcs_bucket}/{self.gcs_output_prefix}")

        return 0 if not failed_members else 1


def main() -> int:
    processor = GRIBToNetCDFProcessor()
    return processor.run()


if __name__ == "__main__":
    exit(main())

