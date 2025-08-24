#!/usr/bin/env python3
"""
GRIB to NetCDF Processor
========================

This script processes AIFS ensemble forecast GRIB files by:
1. Downloading GRIB files from GCS for each member
2. Converting to NetCDF with 1.5° regridding 
3. Uploading NetCDF to GCS in 1p5deg_nc folder
4. Cleaning up local files

Process: member-by-member (1-50), 5 time ranges per member
"""

import os
import time
import tempfile
import shutil
import gc
import glob
import random
from pathlib import Path
from google.cloud import storage
import xarray as xr
import earthkit.data as ekd
import earthkit.regrid as ekr
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


class GRIBToNetCDFProcessor:
    def __init__(self):
        # GCS Configuration
        self.gcs_bucket = "ea_aifs_w1"
        self.gcs_input_prefix = "forecasts/20240821_0000/"
        self.gcs_output_prefix = "1p5deg_nc/"
        self.service_account_key = "/home/sparrow/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json"
        
        # Forecast Configuration - Handle split ensemble structure  
        self.forecast_date = "20250822"
        
        # AIFS ensemble is split by forecast time:
        # 0000: Members 1-28, 1200: Members 29-50
        self.forecast_configs = [
            {"time": "0000", "members": list(range(1, 29))},   # Members 1-28
            {"time": "1200", "members": list(range(29, 51))}   # Members 29-50
        ]
        
        # Time ranges for 792-hour forecast
        self.time_ranges = [
            ("432", "504"),  # Days 18-21
            ("504", "576"),  # Days 21-24
            ("576", "648"),  # Days 24-27
            ("648", "720"),  # Days 27-30
            ("720", "792")   # Days 30-33
        ]
        
        # Variable mapping for NetCDF conversion  
        self.var_mapping = {
            'mslp': 'msl',    # Mean sea level pressure
            'pr': 'tp',       # Total precipitation  
        }
        
        # Temperature will be searched comprehensively in each GRIB file
        # Common temperature variable names to look for
        self.temp_alternatives = ['t2m', '2t', 'skt', 'stl1', 'st']
        
        # GCS client
        self.client = None
        self.bucket = None
        
        # Temporary directory for processing
        self.temp_dir = None
        
    def cleanup_earthkit_databases(self):
        """Clean up any leftover earthkit database locks"""
        try:
            # Common earthkit database locations
            possible_db_paths = [
                os.path.expanduser('~/.earthkit'),
                '/tmp/.earthkit', 
                '/tmp/earthkit',
                tempfile.gettempdir() + '/earthkit'
            ]
            
            for db_path in possible_db_paths:
                if os.path.exists(db_path):
                    # Look for database lock files
                    for root, dirs, files in os.walk(db_path):
                        for file in files:
                            if file.endswith('.lock') or 'lock' in file.lower():
                                lock_file = os.path.join(root, file)
                                try:
                                    os.remove(lock_file)
                                    print(f"    🗑️  Removed earthkit lock: {lock_file}")
                                except:
                                    pass
                                    
        except Exception:
            pass  # Ignore cleanup errors
    
    def initialize_gcs(self):
        """Initialize GCS client and bucket"""
        try:
            print(f"🔗 Initializing GCS connection...")
            
            # Clean up any leftover database locks first
            print(f"🧹 Cleaning up earthkit database locks...")
            self.cleanup_earthkit_databases()
            
            self.client = storage.Client.from_service_account_json(self.service_account_key)
            self.bucket = self.client.bucket(self.gcs_bucket)
            print(f"✅ Connected to GCS bucket: {self.gcs_bucket}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize GCS: {e}")
            return False
    
    def create_temp_directory(self):
        """Create temporary working directory"""
        self.temp_dir = tempfile.mkdtemp(prefix='grib_nc_processor_')
        print(f"📁 Created temporary directory: {self.temp_dir}")
        
    def cleanup_temp_directory(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory")
    
    def process_grib_with_retry(self, grib_file, max_retries=3):
        """Process GRIB file with retry logic for database lock issues"""
        for attempt in range(max_retries):
            try:
                print(f"    🔄 Processing GRIB file (attempt {attempt + 1})...")
                
                # Add random delay to avoid simultaneous database access
                if attempt > 0:
                    delay = random.uniform(1, 5)  # 1-5 second random delay
                    print(f"    ⏳ Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                
                # Process this single GRIB file
                fl = ekd.from_source("file", grib_file)
                
                # Regrid from N320 to 1.5° regular lon/lat
                fl_ll_1p5 = ekr.interpolate(fl, 
                                           in_grid={"grid": "N320"}, 
                                           out_grid={"grid": [1.5, 1.5]})
                
                # Convert to xarray
                ds = fl_ll_1p5.to_xarray()
                
                return ds, fl, fl_ll_1p5  # Return objects for cleanup
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if "database is locked" in error_msg or "database" in error_msg:
                    print(f"    ⚠️  Database lock error (attempt {attempt + 1}): {e}")
                    
                    if attempt < max_retries - 1:
                        # Clean up any partial objects
                        try:
                            if 'fl' in locals(): del fl
                            if 'fl_ll_1p5' in locals(): del fl_ll_1p5  
                            if 'ds' in locals(): del ds
                        except:
                            pass
                        gc.collect()
                        
                        # Longer delay for database locks
                        delay = random.uniform(5, 15)
                        print(f"    ⏳ Database locked, waiting {delay:.1f}s before retry...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"    ❌ Failed after {max_retries} attempts due to database lock")
                        return None, None, None
                else:
                    # Non-database error, don't retry
                    print(f"    ❌ Processing error: {e}")
                    return None, None, None
        
        return None, None, None
    
    def cleanup_earthkit_temp_files(self):
        """Clean up earthkit temporary files and cache"""
        import glob
        import tempfile
        
        try:
            # Clean up system temp directory for earthkit files
            temp_patterns = [
                '/tmp/earthkit*',
                '/tmp/tmp*earthkit*', 
                '/tmp/regrid*',
                '/tmp/*.npz',
                f'{tempfile.gettempdir()}/earthkit*',
                f'{tempfile.gettempdir()}/regrid*',
                f'{tempfile.gettempdir()}/*.npz'
            ]
            
            for pattern in temp_patterns:
                for temp_file in glob.glob(pattern):
                    try:
                        if os.path.isfile(temp_file):
                            os.remove(temp_file)
                        elif os.path.isdir(temp_file):
                            shutil.rmtree(temp_file)
                    except:
                        pass  # Ignore cleanup errors
                        
        except Exception:
            pass  # Ignore any cleanup errors
    
    def force_cleanup_memory_and_disk(self):
        """Aggressive cleanup of memory and disk space"""
        import gc
        import subprocess
        
        try:
            print(f"      🧹 Starting aggressive cleanup...")
            
            # Show space before cleanup
            if psutil:
                disk_usage = psutil.disk_usage('/')
                free_before = disk_usage.free / (1024**3)
                print(f"      💾 Free space before cleanup: {free_before:.1f} GB")
            
            # Force garbage collection multiple times
            for i in range(3):
                gc.collect()
            
            # Clean up earthkit temp files
            self.cleanup_earthkit_temp_files()
            
            # More aggressive temp file cleanup
            temp_dirs_to_clean = ['/tmp', '/var/tmp', tempfile.gettempdir()]
            patterns_to_remove = [
                '*.npz', '*.grib*', '*earthkit*', '*regrid*', 
                'tmp*', '*.tmp', '*.cache', '*mirror*', '*mir_*'
            ]
            
            for temp_dir in temp_dirs_to_clean:
                if os.path.exists(temp_dir):
                    for pattern in patterns_to_remove:
                        full_pattern = os.path.join(temp_dir, pattern)
                        for temp_file in glob.glob(full_pattern):
                            try:
                                if os.path.isfile(temp_file):
                                    file_size = os.path.getsize(temp_file) / (1024*1024)
                                    if file_size > 10:  # Remove files > 10MB
                                        os.remove(temp_file)
                                        print(f"        🗑️  Removed temp file: {os.path.basename(temp_file)} ({file_size:.1f} MB)")
                                elif os.path.isdir(temp_file):
                                    shutil.rmtree(temp_file)
                                    print(f"        🗑️  Removed temp dir: {os.path.basename(temp_file)}")
                            except Exception as e:
                                pass  # Ignore individual cleanup errors
            
            # Use system commands for more thorough cleanup
            try:
                # Clean up any remaining earthkit/regrid cache
                subprocess.run(['find', '/tmp', '-name', '*earthkit*', '-delete'], 
                             capture_output=True, timeout=10)
                subprocess.run(['find', '/tmp', '-name', '*regrid*', '-delete'], 
                             capture_output=True, timeout=10) 
                subprocess.run(['find', '/tmp', '-name', '*.npz', '-size', '+10M', '-delete'], 
                             capture_output=True, timeout=10)
                
                # Clear any xarray cache directories
                subprocess.run(['find', '/tmp', '-name', '*xarray*', '-delete'], 
                             capture_output=True, timeout=10)
                             
            except Exception:
                pass
                    
            # Show space after cleanup
            if psutil:
                disk_usage = psutil.disk_usage('/')
                free_after = disk_usage.free / (1024**3)
                freed_space = free_after - free_before
                print(f"      💾 Free space after cleanup: {free_after:.1f} GB")
                if freed_space > 0:
                    print(f"      ✅ Freed up: {freed_space:.1f} GB")
                else:
                    print(f"      ⚠️  No significant space freed")
            
        except Exception as e:
            print(f"      ⚠️  Cleanup error: {e}")
            pass
    
    def check_disk_space(self, required_gb=2):
        """Check if we have enough disk space"""
        if psutil:
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / (1024**3)
            return free_space_gb > required_gb
        return True  # If psutil not available, assume we have space
    
    def get_forecast_time_for_member(self, member):
        """Get the correct forecast time for a given member"""
        for config in self.forecast_configs:
            if member in config["members"]:
                return config["time"]
        raise ValueError(f"Member {member} not found in any forecast configuration")
    
    def download_grib_file(self, member, time_range):
        """Download a single GRIB file from GCS"""
        # Check disk space before download
        if not self.check_disk_space(2):  # Need at least 2GB free
            print(f"    ⚠️  Insufficient disk space, cleaning up...")
            self.force_cleanup_memory_and_disk()
            
            # Check again after cleanup
            if not self.check_disk_space(2):
                print(f"    ❌ Still insufficient disk space after cleanup")
                return None
        
        # Get correct forecast time for this member
        forecast_time = self.get_forecast_time_for_member(member)
        
        start_hour, end_hour = time_range
        filename = f"aifs_ens_forecast_{self.forecast_date}_{forecast_time}_member{member:03d}_h{start_hour}-{end_hour}.grib"
        blob_name = f"{self.gcs_input_prefix}{filename}"
        local_path = os.path.join(self.temp_dir, filename)
        
        try:
            blob = self.bucket.blob(blob_name)
            if not blob.exists():
                print(f"    ⚠️  File not found: {filename}")
                return None
                
            print(f"    ⬇️  Downloading: {filename}")
            start_time = time.time()
            blob.download_to_filename(local_path)
            
            # Verify download
            file_size = os.path.getsize(local_path)
            download_time = time.time() - start_time
            size_mb = file_size / (1024 * 1024)
            
            print(f"    ✅ Downloaded: {size_mb:.1f} MB in {download_time:.1f}s")
            return local_path
            
        except Exception as e:
            print(f"    ❌ Download failed: {e}")
            return None
    
    def process_grib_to_netcdf(self, member, grib_files):
        """Convert GRIB files to NetCDF with regridding"""
        print(f"  🔄 Converting GRIB to NetCDF...")
        
        try:
            member_datasets = []
            
            for grib_file in grib_files:
                if grib_file is None or not os.path.exists(grib_file):
                    continue
                    
                print(f"    Processing: {os.path.basename(grib_file)}")
                
                try:
                    # Process one GRIB file at a time to manage memory
                    fl = None
                    fl_ll_1p5 = None
                    ds = None
                    
                    try:
                        # Open GRIB file
                        fl = ekd.from_source("file", grib_file)
                        
                        # Regrid from N320 to 1.5° regular lon/lat
                        fl_ll_1p5 = ekr.interpolate(fl, 
                                                   in_grid={"grid": "N320"}, 
                                                   out_grid={"grid": [1.5, 1.5]})
                        
                        # Convert to xarray
                        ds = fl_ll_1p5.to_xarray()
                        
                        # Extract target variables
                        available_vars = list(ds.data_vars)
                        target_vars = []
                        
                        for target, alt_name in self.var_mapping.items():
                            if alt_name in available_vars:
                                target_vars.append(alt_name)
                                print(f"      ✓ Found {alt_name} for {target}")
                        
                        if target_vars:
                            # Extract only target variables and load into memory
                            extracted_ds = ds[target_vars].load()  # Force load to memory
                            member_datasets.append(extracted_ds)
                            print(f"      Extracted: {target_vars}")
                        else:
                            print(f"      ⚠️  No target variables found")
                            
                    finally:
                        # Explicit cleanup of earthkit objects
                        if ds is not None:
                            del ds
                        if fl_ll_1p5 is not None:
                            del fl_ll_1p5
                        if fl is not None:
                            del fl
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Clean up any temporary files created by earthkit after each file
                        self.cleanup_earthkit_temp_files()
                        
                        # Remove the processed GRIB file immediately to free space
                        if grib_file and os.path.exists(grib_file):
                            try:
                                file_size = os.path.getsize(grib_file) / (1024*1024)
                                os.remove(grib_file)
                                print(f"      🗑️  Removed processed GRIB: {os.path.basename(grib_file)} ({file_size:.1f} MB)")
                            except Exception:
                                pass
                    
                except Exception as e:
                    print(f"      ❌ Error processing GRIB: {e}")
                    continue
            
            if not member_datasets:
                print(f"    ⚠️  No valid datasets for member {member:03d}")
                return None
            
            # Concatenate time ranges
            member_combined = xr.concat(member_datasets, dim='time')
            
            # Add member dimension
            member_combined = member_combined.expand_dims('member')
            member_combined = member_combined.assign_coords(member=[f"{member:03d}"])
            
            # Create NetCDF filename
            nc_filename = f"aifs_ensemble_forecast_1p5deg_member{member:03d}.nc"
            nc_path = os.path.join(self.temp_dir, nc_filename)
            
            # Add metadata - dynamically describe all found variables
            available_vars = list(member_combined.data_vars)
            var_descriptions = []
            
            # Create descriptions for all found variables
            for var_name in available_vars:
                if var_name == 'msl':
                    var_descriptions.append('msl (mean sea level pressure)')
                elif var_name == 'tp':
                    var_descriptions.append('tp (total precipitation)')
                elif any(temp_keyword in var_name.lower() for temp_keyword in ['temp', '2m', 't2m', '2t', 'skt', 'skin']):
                    var_descriptions.append(f'{var_name} (temperature)')
                else:
                    # For any other variables found
                    var_descriptions.append(f'{var_name}')
            
            member_combined.attrs.update({
                'title': 'AIFS Ensemble Forecast Data',
                'description': f'Regridded to 1.5 degree resolution, member {member:03d}',
                'source': 'ECMWF AIFS ensemble forecast',
                'grid_resolution': '1.5 degrees',
                'forecast_date': f'{self.forecast_date} {self.forecast_time}:00',
                'member': f'member{member:03d}',
                'variables': ', '.join(var_descriptions),
                'processing_date': str(np.datetime64('now'))
            })
            
            # Clean attributes to avoid NetCDF serialization issues
            member_combined = self.clean_dataset_attrs(member_combined)
            
            # Save to NetCDF
            print(f"    💾 Saving NetCDF: {nc_filename}")
            member_combined.to_netcdf(nc_path, engine='netcdf4')
            
            # Check file size
            nc_size = os.path.getsize(nc_path) / (1024 * 1024)
            print(f"    ✅ NetCDF created: {nc_size:.1f} MB")
            
            return nc_path
            
        except Exception as e:
            print(f"    ❌ NetCDF conversion failed: {e}")
            return None
    
    def clean_dataset_attrs(self, ds):
        """Remove problematic attributes for NetCDF serialization"""
        def clean_attrs(obj):
            if hasattr(obj, 'attrs'):
                attrs_to_remove = []
                for key, value in obj.attrs.items():
                    if key.startswith('_earthkit') or isinstance(value, dict):
                        attrs_to_remove.append(key)
                for key in attrs_to_remove:
                    del obj.attrs[key]
            return obj
        
        # Clean dataset and all variables/coordinates
        ds = clean_attrs(ds)
        for var_name in ds.data_vars:
            ds[var_name] = clean_attrs(ds[var_name])
        for coord_name in ds.coords:
            ds[coord_name] = clean_attrs(ds[coord_name])
            
        return ds
    
    def upload_netcdf_to_gcs(self, nc_path, member):
        """Upload NetCDF file to GCS"""
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
    
    def cleanup_local_files(self, files_to_remove):
        """Remove local files to free up space"""
        for file_path in files_to_remove:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"    🗑️  Removed: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"    ⚠️  Could not remove {file_path}: {e}")
    
    def process_member_streamlined(self, member):
        """Process a single ensemble member with minimal disk usage"""
        print(f"\n{'='*60}")
        print(f"Processing Member {member:03d} (Streamlined)")
        print(f"{'='*60}")
        
        success = False
        nc_file = None
        member_datasets = []
        
        try:
            # Process each time range individually to minimize disk usage
            print(f"🔄 Processing member {member:03d} with minimal disk footprint")
            
            for i, time_range in enumerate(self.time_ranges):
                print(f"  📥 Processing time range {i+1}/{len(self.time_ranges)}: h{time_range[0]}-{time_range[1]}")
                
                # Step 1: Download one GRIB file
                grib_file = self.download_grib_file(member, time_range)
                if grib_file is None:
                    print(f"    ⚠️  Skipping time range h{time_range[0]}-{time_range[1]}")
                    continue
                
                # Step 2: Process GRIB immediately with retry logic
                try:
                    # Use retry logic for database lock issues
                    ds, fl, fl_ll_1p5 = self.process_grib_with_retry(grib_file)
                    
                    if ds is None:
                        print(f"    ❌ Failed to process GRIB file")
                        continue
                    
                    # Extract target variables with comprehensive searching
                    available_vars = list(ds.data_vars)
                    target_vars = []
                    found_temp_var = None
                    
                    print(f"    📋 Available variables: {available_vars}")
                    
                    # Show coordinate info which might contain temperature data
                    print(f"    📋 Dataset coordinates: {list(ds.coords)}")
                    print(f"    📋 Dataset dimensions: {dict(ds.dims)}")
                    
                    # Check for basic variables (msl, tp)
                    for target, alt_name in self.var_mapping.items():
                        if alt_name in available_vars:
                            target_vars.append(alt_name)
                            print(f"    ✓ Found {alt_name} for {target}")
                    
                    # Comprehensive temperature search - check ALL variables
                    temp_keywords = ['temp', '2m', 't2m', '2t', 'skt', 'skin', 'temperature', 'air', 'surface']
                    
                    print(f"    🔍 Searching for temperature variables...")
                    for var_name in available_vars:
                        var_lower = var_name.lower()
                        var_str = str(var_name).lower()
                        
                        # Check if variable name contains temperature keywords
                        is_temp_var = any(keyword in var_lower for keyword in temp_keywords)
                        
                        # Also check variable attributes if available
                        try:
                            if hasattr(ds[var_name], 'long_name'):
                                long_name = str(ds[var_name].long_name).lower()
                                is_temp_var = is_temp_var or any(keyword in long_name for keyword in temp_keywords)
                            
                            if hasattr(ds[var_name], 'standard_name'):
                                standard_name = str(ds[var_name].standard_name).lower()
                                is_temp_var = is_temp_var or any(keyword in standard_name for keyword in temp_keywords)
                        except:
                            pass
                        
                        if is_temp_var and var_name not in target_vars:
                            target_vars.append(var_name)
                            found_temp_var = var_name
                            print(f"    ✓ Found temperature variable: {var_name}")
                            
                            # Show variable details
                            try:
                                var_obj = ds[var_name]
                                print(f"      📊 Shape: {var_obj.shape}")
                                if hasattr(var_obj, 'long_name'):
                                    print(f"      📝 Long name: {var_obj.long_name}")
                                if hasattr(var_obj, 'units'):
                                    print(f"      🎯 Units: {var_obj.units}")
                            except:
                                pass
                            break
                    
                    # If still no temperature found, show all variable details for debugging
                    if not found_temp_var and len(available_vars) > 2:
                        print(f"    🔍 No temperature found, showing details of all variables:")
                        for var_name in available_vars[:5]:  # Show first 5 variables
                            try:
                                var_obj = ds[var_name]
                                print(f"      {var_name}: shape={var_obj.shape}, dims={var_obj.dims}")
                                if hasattr(var_obj, 'long_name'):
                                    print(f"        Long name: {var_obj.long_name}")
                                if hasattr(var_obj, 'standard_name'):
                                    print(f"        Standard name: {var_obj.standard_name}")
                            except Exception as e:
                                print(f"        Error reading {var_name}: {e}")
                    
                    if not found_temp_var:
                        print(f"    ⚠️  No temperature variable found after comprehensive search")
                        print(f"    📋 Available variables: {available_vars}")
                    
                    if target_vars:
                        # Extract and load data immediately
                        extracted_ds = ds[target_vars].load()
                        member_datasets.append(extracted_ds)
                        print(f"    ✅ Extracted: {target_vars}")
                    else:
                        print(f"    ⚠️  No variables could be extracted")
                    
                    # Immediate cleanup of earthkit objects
                    del ds, fl_ll_1p5, fl
                    gc.collect()
                    
                    # Remove GRIB file immediately
                    grib_size = os.path.getsize(grib_file) / (1024*1024)
                    os.remove(grib_file)
                    print(f"    🗑️  Removed GRIB: {grib_size:.1f} MB")
                    
                    # Clean temp files after each GRIB
                    self.cleanup_earthkit_temp_files()
                    
                except Exception as e:
                    print(f"    ❌ Error processing time range: {e}")
                    if grib_file and os.path.exists(grib_file):
                        os.remove(grib_file)
                    continue
            
            # Check if we have any datasets
            if not member_datasets:
                print(f"  ❌ No valid datasets for member {member:03d}")
                return False
            
            # Step 3: Combine datasets and create NetCDF
            print(f"  🔗 Combining {len(member_datasets)} time ranges")
            member_combined = xr.concat(member_datasets, dim='time')
            member_combined = member_combined.expand_dims('member')
            member_combined = member_combined.assign_coords(member=[f"{member:03d}"])
            
            # Add metadata - dynamically describe all found variables
            available_vars = list(member_combined.data_vars)
            var_descriptions = []
            
            # Create descriptions for all found variables
            for var_name in available_vars:
                if var_name == 'msl':
                    var_descriptions.append('msl (mean sea level pressure)')
                elif var_name == 'tp':
                    var_descriptions.append('tp (total precipitation)')
                elif any(temp_keyword in var_name.lower() for temp_keyword in ['temp', '2m', 't2m', '2t', 'skt', 'skin']):
                    var_descriptions.append(f'{var_name} (temperature)')
                else:
                    # For any other variables found
                    var_descriptions.append(f'{var_name}')
            
            member_combined.attrs.update({
                'title': 'AIFS Ensemble Forecast Data',
                'description': f'Regridded to 1.5 degree resolution, member {member:03d}',
                'source': 'ECMWF AIFS ensemble forecast',
                'grid_resolution': '1.5 degrees',
                'forecast_date': f'{self.forecast_date} {self.forecast_time}:00',
                'member': f'member{member:03d}',
                'variables': ', '.join(var_descriptions),
                'processing_date': str(np.datetime64('now'))
            })
            
            # Clean attributes
            member_combined = self.clean_dataset_attrs(member_combined)
            
            # Save to NetCDF
            nc_filename = f"aifs_ensemble_forecast_1p5deg_member{member:03d}.nc"
            nc_file = os.path.join(self.temp_dir, nc_filename)
            
            print(f"  💾 Saving NetCDF: {nc_filename}")
            member_combined.to_netcdf(nc_file, engine='netcdf4')
            
            nc_size = os.path.getsize(nc_file) / (1024 * 1024)
            print(f"  ✅ NetCDF created: {nc_size:.1f} MB")
            
            # Clear the datasets from memory
            del member_datasets, member_combined
            gc.collect()
            
            # Step 4: Upload NetCDF to GCS
            print(f"  ☁️  Uploading NetCDF to GCS")
            upload_success = self.upload_netcdf_to_gcs(nc_file, member)
            
            if not upload_success:
                print(f"  ❌ Upload failed for member {member:03d}")
                return False
            
            success = True
            print(f"  ✅ Member {member:03d} processed successfully!")
            
        except Exception as e:
            print(f"  ❌ Error processing member {member:03d}: {e}")
            success = False
        
        finally:
            # Final cleanup
            print(f"  🧹 Final cleanup")
            if nc_file and os.path.exists(nc_file):
                try:
                    os.remove(nc_file)
                    print(f"    🗑️  Removed NetCDF file")
                except:
                    pass
            
            # Aggressive cleanup
            self.force_cleanup_memory_and_disk()
        
        return success
    
    def run(self):
        """Main processing pipeline"""
        print(f"GRIB to NetCDF Processor")
        print(f"{'='*60}")
        print(f"Bucket: {self.gcs_bucket}")
        print(f"Input path: {self.gcs_input_prefix}")
        print(f"Output path: {self.gcs_output_prefix}")
        # Calculate total members from both forecast times
        all_members = []
        for config in self.forecast_configs:
            all_members.extend(config["members"])
        
        print(f"Members: {min(all_members)}-{max(all_members)} (split across forecast times)")
        print(f"  0000 forecast: Members {min(self.forecast_configs[0]['members'])}-{max(self.forecast_configs[0]['members'])} ({len(self.forecast_configs[0]['members'])} members)")
        print(f"  1200 forecast: Members {min(self.forecast_configs[1]['members'])}-{max(self.forecast_configs[1]['members'])} ({len(self.forecast_configs[1]['members'])} members)")
        print(f"Time ranges: {len(self.time_ranges)} periods")
        print(f"Forecast date: {self.forecast_date}")
        
        # Initialize
        if not self.initialize_gcs():
            return 1
            
        self.create_temp_directory()
        
        # Track progress
        successful_members = []
        failed_members = []
        start_time = time.time()
        
        try:
            # Process all members from both forecast times
            all_members = []
            for config in self.forecast_configs:
                all_members.extend(config["members"])
            
            # Process each member
            for i, member in enumerate(all_members):
                forecast_time = self.get_forecast_time_for_member(member)
                print(f"\n🔄 Processing member {i+1}/{len(all_members)} (Member {member:03d} from {forecast_time} forecast)")
                
                member_start_time = time.time()
                success = self.process_member_streamlined(member)  # Use streamlined version
                member_time = time.time() - member_start_time
                
                if success:
                    successful_members.append(member)
                else:
                    failed_members.append(member)
                
                # Progress estimate
                elapsed_total = time.time() - start_time
                if i > 0:
                    avg_time_per_member = elapsed_total / (i + 1)
                    remaining_time = avg_time_per_member * (len(all_members) - i - 1)
                    print(f"  ⏱️  Member time: {member_time/60:.1f} min")
                    print(f"  ⏱️  Estimated remaining: {remaining_time/60:.1f} min")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Processing interrupted by user")
        
        finally:
            self.cleanup_temp_directory()
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        total_members = len(all_members) if 'all_members' in locals() else 0
        print(f"✅ Successful: {len(successful_members)}/{total_members} members")
        print(f"❌ Failed: {len(failed_members)} members")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        
        if failed_members:
            print(f"\nFailed members: {failed_members}")
        
        if successful_members:
            print(f"\nNetCDF files uploaded to: gs://{self.gcs_bucket}/{self.gcs_output_prefix}")
        
        return 0 if len(failed_members) == 0 else 1


def main():
    """Entry point"""
    processor = GRIBToNetCDFProcessor()
    return processor.run()


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)