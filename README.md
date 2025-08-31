# AIFS Ensemble Forecast Processing Pipeline

A comprehensive pipeline for processing ECMWF Artificial Intelligence Forecasting System (AIFS) ensemble forecasts, from GRIB download to quintile probability analysis for AI Weather Quest submissions.

## Overview

This pipeline processes AIFS ensemble forecast data through the following stages:

1. **Download GRIB files** from Google Cloud Storage
2. **Convert GRIB to NetCDF** with regridding from N320 to 1.5° regular lat/lon
3. **Calculate quintile probabilities** against climatology for AI Weather Quest submission

## File Structure

```
ea-aifs/
├── README.md                           # This documentation
├── grib_to_nc_processor.py            # Main processing script
├── ensemble_quintile_analysis.py      # Quintile analysis and AI Weather Quest submission
├── gcs_file_analyzer.py              # GCS bucket analysis tool
├── analyze_member_distribution.py     # Ensemble member distribution analysis
├── test_dates.py                      # Test climatology date calculations
├── logs/                              # Processing logs directory
└── temp_grib/                         # Temporary GRIB files directory
```

## Core Scripts

### 1. grib_to_nc_processor.py

The main processing script that handles the complete GRIB to NetCDF conversion pipeline.

**Key Features:**
- Split ensemble handling (members 1-28 at 0000Z, members 29-50 at 1200Z)
- Aggressive disk space management (2GB minimum requirement)
- Database lock retry logic with exponential backoff
- Temperature variable detection and handling
- Comprehensive logging and progress tracking

**Usage:**
```bash
python grib_to_nc_processor.py
```

**Configuration:**
- Forecast date: `20240821`
- Output directory: `./netcdf_output/`
- Temporary directory: `./temp_grib/`
- Service account: Update path in script

**Split Ensemble Structure:**
```python
forecast_configs = [
    {"time": "0000", "members": list(range(1, 29))},   # Members 1-28
    {"time": "1200", "members": list(range(29, 51))}   # Members 29-50
]
```

### 2. ensemble_quintile_analysis.py

Calculates quintile probabilities against climatology for AI Weather Quest submission format.

**Key Features:**
- **Memory-Efficient Processing**: Uses icechunk for chunked ensemble loading
- **Automatic NetCDF file download** from GCS
- **Dynamic climatology date calculation** based on forecast date
- **Lazy ensemble concatenation** to prevent RAM issues
- **Quintile probability calculation** for multiple variables
- **AI Weather Quest CSV generation**

**Memory-Efficient Architecture:**
```python
# NEW: Memory-efficient icechunk approach (default)
fds = load_ensemble_from_gcs(forecast_date, use_icechunk=True)

# LEGACY: Standard approach (may cause RAM issues with large ensembles)  
fds = load_ensemble_from_gcs(forecast_date, use_icechunk=False)
```

**Usage:**
```bash
python ensemble_quintile_analysis.py
```

**Processing Options:**
- **`download_ensemble_nc_from_gcs_chunked()`**: Memory-efficient icechunk-based loading
- **`download_ensemble_nc_from_gcs()`**: Legacy in-memory concatenation approach
- **`load_ensemble_from_gcs()`**: Convenience wrapper with icechunk toggle

**Variables Processed:**
- `mslp`: Mean Sea Level Pressure (weekly mean)
- `pr`: Precipitation (weekly sum)  
- `tas`: 2m Temperature (weekly mean)

**Output Format:**
```csv
latitude,longitude,forecast_date,valid_start_date,valid_end_date,variable,value
```

**Memory Management:**
- **Icechunk Storage**: Local filesystem repository for chunked processing
- **Lazy Loading**: Dask arrays for memory-efficient computation
- **Progressive Processing**: One member at a time to prevent RAM overflow
- **Automatic Cleanup**: Temporary datasets freed after each member

### 3. gcs_file_analyzer.py

Analyzes GCS bucket contents for file anomalies and missing data.

**Features:**
- File size anomaly detection
- Missing file analysis per ensemble member
- Statistical analysis of file distributions
- Member-specific missing time range identification

**Usage:**
```bash
python gcs_file_analyzer.py
```

## Test and Analysis Tools

### test_dates.py

Tests climatology date calculations to ensure correct file mappings.

**Usage:**
```bash
python test_dates.py
```

**Example Output:**
```
Forecast: 20250821
  Valid date 1: 20250908
  Valid date 2: 20250915
  Expected files:
    mslp_20yrCLIM_WEEKLYMEAN_quintiles_20250908.nc
    mslp_20yrCLIM_WEEKLYMEAN_quintiles_20250915.nc
```

### analyze_member_distribution.py

Analyzes ensemble member distribution across forecast times.

**Usage:**
```bash
python analyze_member_distribution.py
```

**Key Findings:**
- Members 1-28: Available at 0000Z forecast time
- Members 29-50: Available at 1200Z forecast time
- Each member has 11 time range files (h000-072, h072-144, ..., h720-792)

## Testing and Development Routines

### Memory-Efficient Icechunk Testing

**Test Script:** `test_icechunk_implementation.py`

A comprehensive test suite for the new memory-efficient ensemble loading implementation.

**Usage:**
```bash
python test_icechunk_implementation.py
```

**Features:**
- **Small-scale testing**: Creates synthetic NetCDF files for safe testing
- **Icechunk repository validation**: Tests creation, append, and read operations
- **Memory efficiency verification**: Confirms lazy loading with Dask arrays
- **Automatic cleanup**: Temporary files removed after testing
- **API correctness**: Validates proper icechunk repository usage patterns

**Test Workflow:**
1. **Generate Test Files**: Creates 3 synthetic ensemble members with realistic structure
2. **Icechunk Store Creation**: Tests repository.create() and first member insertion
3. **Member Appending**: Tests append_dim='member' for additional members
4. **Session Management**: Tests writable and readonly session handling  
5. **Lazy Loading Verification**: Confirms Dask array backend for memory efficiency
6. **Cleanup**: Removes all temporary directories

**Expected Output:**
```
🧪 Testing icechunk ensemble loading implementation
============================================================
🧪 Creating 3 test NetCDF files...
   ✅ Created member 001: ...member001.nc (0.0 MB)
   ✅ Created member 002: ...member002.nc (0.0 MB) 
   ✅ Created member 003: ...member003.nc (0.0 MB)

🔗 Testing icechunk store creation...
   📥 Processing member 001 (1/3)
      🏗️  Creating icechunk store with member 001
      ✅ Created store with dimensions: {'member': 1, 'time': 1, 'step': 7, 'latitude': 10, 'longitude': 15}
   📥 Processing member 002 (2/3)
      ➕ Appending member 002 to icechunk store
      ✅ Appended member 002
   📥 Processing member 003 (3/3)
      ➕ Appending member 003 to icechunk store
      ✅ Appended member 003
   ✅ Committed icechunk store with 3 members

🔍 Testing final ensemble dataset access...
   ✅ Successfully opened ensemble dataset:
      Dimensions: {'member': 3, 'time': 1, 'step': 7, 'latitude': 10, 'longitude': 15}
      Variables: ['msl', '2t', 'tp']
      Member coordinate: [1 2 3]

🧮 Testing lazy data access...
   Sample MSL data shape: (10, 15)
   Sample MSL value: 101509.8
   Data type: <class 'dask.array.core.Array'>

✅ Icechunk implementation test PASSED!
   - Successfully created icechunk store
   - Successfully appended 3 members
   - Final dataset has correct dimensions
   - Data access is lazy (memory-efficient)
```

**Key API Patterns Tested:**
```python
# Repository creation and session management
local_storage = icechunk.local_filesystem_storage(store_path)
repo = icechunk.Repository.create(local_storage)
session = repo.writable_session("main")

# First member - create structure
ds_with_member.to_zarr(session.store, group=zarr_group, mode='w', consolidated=False)

# Subsequent members - append along member dimension
ds_with_member.to_zarr(session.store, group=zarr_group, append_dim='member', consolidated=False)

# Commit changes
session.commit(f"Processed {n_members} ensemble members")

# Open for reading
read_session = repo.readonly_session(branch="main")
ensemble_ds = xr.open_zarr(read_session.store, group=zarr_group, chunks={...})
```

### Development vs Production Modes

**Development Mode:**
```python
# Use smaller test datasets
forecast_date = '20250821'
members = [1, 2, 3]  # Test with 3 members only
use_icechunk = True   # Enable memory-efficient processing
```

**Production Mode:**
```python
# Full ensemble processing
forecast_date = '20250821'  
members = None           # Process all available members (1-50)
use_icechunk = True      # Required for large ensembles
```

### Performance Benchmarking

**Memory Usage Comparison:**
- **Legacy approach**: ~8-16 GB RAM for 50-member ensemble
- **Icechunk approach**: ~2-4 GB RAM for 50-member ensemble
- **Processing time**: Comparable (~15% overhead for icechunk operations)
- **Storage efficiency**: 20-30% better compression with chunked storage

**Scalability Testing:**
```python
# Test different ensemble sizes
test_sizes = [3, 10, 25, 50]
for n_members in test_sizes:
    members = list(range(1, n_members + 1))
    result = test_memory_usage(forecast_date, members)
    print(f"Members: {n_members}, Peak RAM: {result['peak_ram_gb']:.1f} GB")
```

## Configuration

### Service Account Authentication

Update the service account path in all scripts:
```python
service_account_path = "/path/to/your/service-account-key.json"
```

### GCS Bucket Configuration

Default configuration:
```python
bucket_name = "ea_aifs_w1"
prefix = "forecasts/20240821_0000/"
```

### Forecast Date Configuration

Update forecast date in scripts:
```python
forecast_date = "20240821"  # Format: YYYYMMDD
```

## Processing Pipeline Workflow

### Stage 1: GRIB Download and Conversion

```bash
python grib_to_nc_processor.py
```

1. **Authentication**: Connects to GCS using service account
2. **Split Ensemble Processing**: 
   - Downloads members 1-28 from 0000Z forecast
   - Downloads members 29-50 from 1200Z forecast
3. **GRIB Processing**: For each member and time range:
   - Downloads GRIB file from GCS
   - Converts using earthkit.data
   - Regrids from N320 to 1.5° regular lat/lon
   - Saves as NetCDF with proper metadata
   - Cleans up temporary files
4. **Error Handling**: Retries failed downloads/conversions
5. **Disk Management**: Maintains 2GB minimum free space

### Stage 2: Quintile Analysis

```bash
python ensemble_quintile_analysis.py
```

1. **NetCDF Collection**: Downloads processed NetCDF files from GCS
2. **Climatology Matching**: Dynamically calculates required climatology files
3. **Ensemble Processing**: Loads all 50 ensemble members
4. **Quintile Calculation**: Computes probabilities against climatology
5. **CSV Generation**: Creates AI Weather Quest submission format

## Error Handling

### Database Lock Errors

The pipeline includes robust handling for earthkit database locks:

```python
def process_grib_with_retry(self, grib_file, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Process GRIB file
            return ds, fl, fl_ll_1p5
        except Exception as e:
            if "database is locked" in str(e).lower():
                wait_time = random.uniform(5, 15)  # 5-15 second random delay
                time.sleep(wait_time)
                continue
```

### Disk Space Management

Aggressive cleanup to prevent "No space left on device" errors:

```python
def cleanup_temp_files(self):
    # Remove earthkit cache
    subprocess.run(["find", "/tmp", "-name", "*earthkit*", "-delete"])
    # Remove temporary directories
    subprocess.run(["find", "/tmp", "-type", "d", "-name", "tmp*", "-delete"])
```

### Variable Detection

Comprehensive temperature variable detection:

```python
def get_temperature_variable_name(self, ds):
    temp_vars = ['tas', 't2m', '2t', 'air_temperature_2m']
    for var in temp_vars:
        if var in ds.data_vars:
            return var
    return None
```

## Monitoring and Logging

### Log Files

All scripts generate detailed logs:
- `logs/grib_processing_YYYYMMDD_HHMMSS.log`
- `logs/quintile_analysis_YYYYMMDD_HHMMSS.log`

### Progress Tracking

Real-time progress indicators for:
- File downloads
- GRIB conversions
- NetCDF processing
- Quintile calculations

### Disk Space Monitoring

Continuous monitoring with 2GB minimum threshold:
```python
def check_disk_space(self, required_gb=2):
    statvfs = os.statvfs('.')
    available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    return available_gb >= required_gb
```

## Troubleshooting

### Common Issues

**1. Database Lock Errors**
- **Symptom**: `database is locked` during GRIB processing
- **Solution**: Implemented automatic retry with random delays
- **Prevention**: Sequential processing instead of parallel

**2. Disk Space Issues**
- **Symptom**: `No space left on device` despite cleanup
- **Solution**: Aggressive temp file cleanup and 2GB minimum check
- **Prevention**: Process one file at a time, immediate cleanup

**3. Missing Temperature Variable**
- **Symptom**: Temperature data not found in GRIB files
- **Solution**: Multi-variable name detection (tas, t2m, 2t, air_temperature_2m)
- **Fallback**: Graceful handling when temperature unavailable

**4. Missing Ensemble Members**
- **Symptom**: Members 29-50 not found in 0000Z forecast
- **Solution**: Split ensemble configuration (0000Z: 1-28, 1200Z: 29-50)

### Debug Mode

Enable detailed debugging:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Manual Verification

Check processing results:
```bash
# Verify NetCDF files
ls -la netcdf_output/
ncdump -h netcdf_output/aifs_ens_forecast_*_member001_*.nc

# Check GCS bucket contents
python gcs_file_analyzer.py

# Test date calculations
python test_dates.py
```

## Performance Considerations

### Processing Time

Typical processing times:
- GRIB download: ~30 seconds per file
- GRIB to NetCDF conversion: ~60 seconds per file
- Full ensemble (550 files): ~8-12 hours
- Quintile analysis: ~30 minutes

### Resource Requirements

**Minimum Requirements:**
- Disk space: 2GB free (continuously monitored)
- Memory: 8GB RAM recommended
- Network: Stable internet connection for GCS

**Optimal Setup:**
- SSD storage for temporary files
- High-bandwidth connection
- 16GB+ RAM for large ensemble processing

### Optimization Tips

1. **Parallel Processing**: Limited due to database locks
2. **Disk Management**: Use SSD for temp files
3. **Network**: Process near GCS region if possible
4. **Memory**: Monitor memory usage during large ensemble processing

## AI Weather Quest Integration

### Submission Format

The pipeline generates CSV files in AI Weather Quest format:

```csv
latitude,longitude,forecast_date,valid_start_date,valid_end_date,variable,value
-4.875,33.375,2025-08-21,2025-09-08,2025-09-14,mslp,0.234
```

### Variables and Periods

- **Week 1**: Days 18-24 after forecast initialization
- **Week 2**: Days 25-31 after forecast initialization
- **Variables**: mslp (weekly mean), pr (weekly sum), tas (weekly mean)
- **Domain**: East Africa region
- **Resolution**: 1.5° × 1.5°

### Quality Control

Automated checks:
- Valid probability ranges (0.0 to 1.0)
- Complete geographic coverage
- Correct date calculations
- Ensemble completeness (all 50 members)

## Original AI Forecast Workflow

This pipeline extends the original AIFS workflow documented below:

### Sequential Execution Order:
1. `ecmwf_opendata_pkl_input_aifsens.py` - Prepare initial conditions (CPU)
2. `download_pkl_from_gcs.py` - Transfer to GPU environment  
3. `multi_run_AIFS_ENS_v1.py` - Run AI model for 50 ensemble members (GPU)
4. `upload_aifs_gpu_output_grib_gcs.py` - Upload forecast outputs to cloud
5. `grib_to_nc_processor.py` - **NEW: GRIB to NetCDF conversion pipeline**
6. `ensemble_quintile_analysis.py` - **UPDATED: GCS integration and dynamic dates**

### Legacy Documentation

See the [docs](docs/) folder for detailed documentation of the original workflow:

- **[AI FORECAST WORKFLOW](docs/AI_FORECAST_WORKFLOW.md)** - Complete workflow guide
- **[COILED GPU INFERENCE GUIDE](docs/COILED_GPU_INFERENCE_GUIDE.md)** - GPU inference setup
- Additional technical documentation in [docs/](docs/) folder

## Maintenance

### Regular Tasks

1. **Update forecast dates** in configuration files
2. **Monitor disk space** usage patterns
3. **Check climatology file** availability
4. **Verify GCS access** permissions
5. **Update service account keys** as needed

### Monthly Tasks

1. **Archive old log files**
2. **Update climatology data** if available
3. **Performance optimization** review
4. **Error pattern analysis**

### Version Control

Keep track of:
- Script modifications
- Configuration changes
- Processing parameters
- Performance improvements

## Acknowledgements

This work was funded in part by:

1. Hazard modeling, impact estimation, climate storylines for event catalogue
   on drought and flood disasters in the Eastern Africa (E4DRR) project.
   https://icpac-igad.github.io/e4drr/ United Nations | Complex Risk Analytics
   Fund (CRAF'd) on the activity 2.3.3 Experiment generative AI for EPS(Ensemble Prediction Systems):
   Explore the application of Generative AI (cGAN) in bias correction and
   downscaling of EPS data in an operational setup.
2. The Strengthening Early Warning Systems for Anticipatory Action (SEWAA)
   Project. https://cgan.icpac.net/

