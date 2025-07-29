# ea-aifs
ECMWF AIFS ensemble model run routines.

## Documentation

See the [docs](docs/) folder for detailed documentation:

- [CONTEXT EXPLANATION](docs/context_explanation.md)
- [DETAILED GRIB SAVING EXPLANATION](docs/detailed_grib_saving_explanation.md)
- [DETAILED NETCDF SAVING EXPLANATION](docs/detailed_netcdf_saving_explanation.md)
- [EARTHKIT DATA USAGE EXPLANATION](docs/earthkit_data_usage_explanation.md)
- [FORECAST OUTPUTS DOCUMENTATION](docs/forecast_outputs_documentation.md)

## Python Scripts

- `forecast_with_outputs.py` - Forecast script with GRIB and NetCDF output capabilities
- `multi_run_AIFS_ENS_v1.py` - Multi-run AIFS ENS v1 for 50 ensemble members
- `simple_netcdf_saving_example.py` - Simple example for NetCDF saving
- `test_ensemble_input_states.py` - Test script for ensemble input states

## Jupyter Notebooks

- `multi_run_AIFS_ENS_v1.ipynb` - Interactive notebook for multi-run AIFS ensemble
- `run_AIFS_ENS_v1.ipynb` - Interactive notebook for single AIFS ensemble run

## Docker Environment

The project includes a Docker environment in the `env/` folder for running AIFS models with CUDA support. See [docs/](docs/) for Docker setup and Coiled integration notes. 
