# AIFS ENS v1 Forecast Output Extensions

This documentation describes the extensions added to the AIFS ENS v1 forecast notebook to save forecast results to GRIB and NetCDF files.

## Overview

The original notebook (`run_AIFS_ENS_v1.ipynb`) demonstrated how to run ECMWF's AIFS ENS v1 model using the `anemoi-inference` package. The extensions add functionality to save forecast states to standardized meteorological file formats.

## Files Created

### 1. `forecast_with_outputs.py`
A standalone Python script that provides a complete forecast workflow with file output capabilities.

**Key features:**
- `ForecastRunner` class that encapsulates the entire forecast workflow
- Automatic data retrieval from ECMWF Open Data
- Configurable ensemble member selection
- Simultaneous GRIB and NetCDF output
- Memory optimization settings
- Error handling and progress reporting

**Usage:**
```python
# Run the script directly
python forecast_with_outputs.py

# Or use the ForecastRunner class
from forecast_with_outputs import ForecastRunner

runner = ForecastRunner({"huggingface": "ecmwf/aifs-ens-1.0"})
input_state = runner.prepare_input_state()
forecast_states = runner.run_forecast(
    input_state=input_state,
    lead_time=72,
    grib_output="forecast.grib",
    netcdf_output="forecast.nc"
)
```

### 2. Extended Notebook Cells
The original notebook has been extended with new cells that demonstrate:
- Setting up output directories and file paths
- Configuring NetCDF and GRIB output objects
- Running forecasts with file outputs
- Verifying output files

## Output Formats

### GRIB Format
- **File extension**: `.grib`
- **Use case**: Standard meteorological data exchange format
- **Features**: Compact, widely supported by meteorological software
- **Implementation**: Uses `anemoi.inference.outputs.gribfile.GribFileOutput`

### NetCDF Format
- **File extension**: `.nc`
- **Use case**: Scientific data storage and analysis
- **Features**: Self-describing, supports metadata, works with xarray/pandas
- **Implementation**: Uses `anemoi.inference.outputs.netcdf.NetCDFOutput`

## Configuration Options

### Context Parameters
```python
context = Context()
context.time_step = 6        # Hours between forecast steps
context.lead_time = 72       # Total forecast duration (hours)
context.reference_date = DATE # Reference date for forecast
```

### Output Parameters
Both output classes support:
- `path`: Output file path
- `variables`: List of variables to save (optional, defaults to all)
- `output_frequency`: Frequency of output steps (optional)
- `write_initial_state`: Whether to include initial conditions (optional)

### NetCDF-specific Parameters
- `float_size`: Data precision ("f4" for 32-bit, "f8" for 64-bit)
- `missing_value`: Value to use for missing data

### GRIB-specific Parameters
- `encoding`: GRIB encoding parameters
- `templates`: GRIB template configurations
- `split_output`: Whether to split output into separate files

## Memory Optimization

For large forecasts, set these environment variables:
```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['ANEMOI_INFERENCE_NUM_CHUNKS'] = '16'
```

## Workflow Steps

1. **Data Retrieval**: Fetch initial conditions from ECMWF Open Data
2. **State Preparation**: Convert raw data to model input format
3. **Model Loading**: Initialize the AIFS ENS v1 model
4. **Output Setup**: Configure GRIB and NetCDF output objects
5. **Forecast Execution**: Run the forecast with real-time file writing
6. **File Verification**: Check output files and display metadata

## Example Output Files

### Generated Files
- `forecast_outputs/aifs_ens_forecast_YYYYMMDD_HHMM.grib`
- `forecast_outputs/aifs_ens_forecast_YYYYMMDD_HHMM.nc`

### File Contents
Both files contain the same meteorological fields:
- **Surface fields**: 10m u/v wind, 2m temperature, sea level pressure, etc.
- **Pressure level fields**: Geopotential, temperature, u/v wind, humidity, etc.
- **Soil fields**: Soil temperature and moisture at multiple levels

### Coordinate Information
- **Spatial**: Latitude/longitude coordinates for each grid point
- **Temporal**: Time stamps for each forecast step
- **Vertical**: Pressure levels for atmospheric variables

## Error Handling

The implementation includes error handling for:
- Missing data from ECMWF Open Data API
- Model initialization failures
- File writing permissions
- Memory allocation issues

## Integration with Existing Code

The extensions are designed to be non-intrusive:
- Original notebook functionality remains unchanged
- New cells can be run independently
- Backward compatibility maintained
- Optional output parameters

## Performance Considerations

### File Size Expectations
- **GRIB files**: ~50-200 MB for 72-hour forecasts
- **NetCDF files**: ~100-400 MB for 72-hour forecasts
- Size depends on number of variables, forecast length, and compression

### Memory Usage
- Peak memory usage occurs during model initialization
- File writing is incremental and memory-efficient
- Consider using GPU memory optimization for large forecasts

## Dependencies

Additional packages required for file output:
- `netCDF4`: For NetCDF file creation
- `xarray`: For NetCDF file verification (optional)
- `earthkit-data`: For GRIB file verification (optional)

## Future Extensions

Potential enhancements:
- Compression options for NetCDF files
- Custom GRIB encoding templates
- Parallel processing for ensemble forecasts
- Cloud storage integration
- Visualization tools for output files

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure write permissions for output directory
2. **Memory Errors**: Reduce `ANEMOI_INFERENCE_NUM_CHUNKS` or use CPU
3. **Missing Dependencies**: Install required packages for file formats
4. **Data Retrieval Failures**: Check ECMWF Open Data API availability

### File Verification

Use the verification cells in the notebook to check:
- File existence and size
- Metadata consistency
- Data integrity
- Coordinate accuracy

## References

- [Anemoi Inference Documentation](https://anemoi-inference.readthedocs.io/)
- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [NetCDF Documentation](https://unidata.github.io/netcdf4-python/)
- [GRIB Format Specification](https://www.wmo.int/pages/prog/www/WMOCodes/Guides/GRIB/GRIB2_062006.pdf)