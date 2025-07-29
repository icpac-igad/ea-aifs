#!/usr/bin/env python3
"""
Forecast script with GRIB and NetCDF output capabilities.

This script extends the notebook functionality to run AIFS ENS v1 forecasts
and save the results to both GRIB and NetCDF formats.
"""

import datetime
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
from ecmwf.opendata import Client as OpendataClient

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
from anemoi.inference.outputs.netcdf import NetCDFOutput
from anemoi.inference.outputs.gribfile import GribFileOutput
from anemoi.inference.context import Context


class ForecastRunner:
    """Main forecast runner class that handles data retrieval, forecasting, and output."""
    
    def __init__(self, checkpoint: Dict[str, str], device: str = "cuda"):
        """Initialize the forecast runner.
        
        Parameters
        ----------
        checkpoint : Dict[str, str]
            Model checkpoint configuration
        device : str, optional
            Device to run the model on, by default "cuda"
        """
        self.checkpoint = checkpoint
        self.device = device
        self.runner = None
        
        # ECMWF parameters configuration
        self.PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
        self.PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
        self.PARAM_SOIL = ["sot"]
        self.PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
        self.LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        self.SOIL_LEVELS = [1, 2]
        
    def get_open_data(self, param: List[str], levelist: List[int] = None, 
                      number: Optional[int] = None, date: Optional[datetime.datetime] = None) -> Dict:
        """Retrieve data from ECMWF Open Data API.
        
        Parameters
        ----------
        param : List[str]
            List of parameters to retrieve
        levelist : List[int], optional
            List of levels to retrieve, by default None
        number : Optional[int], optional
            Ensemble number, by default None
        date : Optional[datetime.datetime], optional
            Date for forecast, by default None (uses latest)
            
        Returns
        -------
        Dict
            Dictionary of fields with their values
        """
        if date is None:
            date = OpendataClient("ecmwf").latest()
            
        fields = defaultdict(list)
        
        # Get data for current date and previous date
        for forecast_date in [date - datetime.timedelta(hours=6), date]:
            if number is None:
                data = ekd.from_source("ecmwf-open-data", date=forecast_date, param=param, levelist=levelist)
            else:
                data = ekd.from_source("ecmwf-open-data", date=forecast_date, param=param, 
                                     levelist=levelist, number=[number], stream='enfo')
            
            for f in data:
                # Open data is between -180 and 180, shift to 0-360
                assert f.to_numpy().shape == (721, 1440)
                values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
                
                # Interpolate from 0.25 to N320
                values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
                
                # Create field name
                name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
                fields[name].append(values)
        
        # Stack values for each parameter
        for param, values in fields.items():
            fields[param] = np.stack(values)
            
        return fields
        
    def prepare_input_state(self, date: Optional[datetime.datetime] = None, 
                           ensemble_number: Optional[int] = None) -> Dict:
        """Prepare input state for forecast.
        
        Parameters
        ----------
        date : Optional[datetime.datetime], optional
            Forecast date, by default None (uses latest)
        ensemble_number : Optional[int], optional
            Ensemble member number (1-50), by default None (control)
            
        Returns
        -------
        Dict
            Input state dictionary
        """
        if date is None:
            date = OpendataClient("ecmwf").latest()
            
        print(f"Preparing input state for date: {date}")
        
        fields = {}
        
        # Get surface fields
        print("Retrieving surface fields...")
        fields.update(self.get_open_data(param=self.PARAM_SFC, number=ensemble_number, date=date))
        fields.update(self.get_open_data(param=self.PARAM_SFC_FC, date=date))
        
        # Get soil fields
        print("Retrieving soil fields...")
        soil = self.get_open_data(param=self.PARAM_SOIL, levelist=self.SOIL_LEVELS, 
                                 number=ensemble_number, date=date)
        
        # Rename soil parameters to match model training
        mapping = {'sot_1': 'stl1', 'sot_2': 'stl2', 'vsw_1': 'swvl1', 'vsw_2': 'swvl2'}
        for k, v in soil.items():
            if k in mapping:
                fields[mapping[k]] = v
                
        # Get pressure level fields
        print("Retrieving pressure level fields...")
        fields.update(self.get_open_data(param=self.PARAM_PL, levelist=self.LEVELS, 
                                        number=ensemble_number, date=date))
        
        # Convert geopotential height to geopotential
        print("Converting geopotential height to geopotential...")
        for level in self.LEVELS:
            if f"gh_{level}" in fields:
                gh = fields.pop(f"gh_{level}")
                fields[f"z_{level}"] = gh * 9.80665
                
        return {"date": date, "fields": fields}
        
    def run_forecast(self, input_state: Dict, lead_time: int, 
                    grib_output: Optional[str] = None, 
                    netcdf_output: Optional[str] = None,
                    print_progress: bool = True) -> List[Dict]:
        """Run forecast with optional file outputs.
        
        Parameters
        ----------
        input_state : Dict
            Input state dictionary
        lead_time : int
            Lead time in hours
        grib_output : Optional[str], optional
            Path to save GRIB file, by default None
        netcdf_output : Optional[str], optional
            Path to save NetCDF file, by default None
        print_progress : bool, optional
            Whether to print progress, by default True
            
        Returns
        -------
        List[Dict]
            List of forecast states
        """
        if self.runner is None:
            print("Initializing model runner...")
            self.runner = SimpleRunner(self.checkpoint, device=self.device)
            
        # Create context for outputs
        context = Context()
        context.time_step = 6  # 6-hour time step
        context.lead_time = lead_time
        context.reference_date = input_state["date"]
        
        # Initialize outputs
        outputs = []
        
        if grib_output:
            print(f"Setting up GRIB output: {grib_output}")
            grib_out = GribFileOutput(context, path=grib_output)
            outputs.append(grib_out)
            
        if netcdf_output:
            print(f"Setting up NetCDF output: {netcdf_output}")
            netcdf_out = NetCDFOutput(context, path=netcdf_output)
            outputs.append(netcdf_out)
            
        # Run forecast
        print(f"Running forecast with lead time: {lead_time} hours...")
        forecast_states = []
        
        for i, state in enumerate(self.runner.run(input_state=input_state, lead_time=lead_time)):
            if print_progress:
                print_state(state)
                
            # Initialize outputs on first state
            if i == 0:
                for output in outputs:
                    output.open(state)
                    
            # Write to outputs
            for output in outputs:
                output.write_step(state)
                
            forecast_states.append(state)
            
        # Close outputs
        for output in outputs:
            output.close()
            
        print(f"Forecast completed. Generated {len(forecast_states)} time steps.")
        
        if grib_output:
            print(f"GRIB file saved: {grib_output}")
        if netcdf_output:
            print(f"NetCDF file saved: {netcdf_output}")
            
        return forecast_states


def main():
    """Main function to run the forecast."""
    # Configuration
    checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
    device = "cuda"
    lead_time = 72  # 72 hours (3 days)
    
    # Optional: Set environment variables for memory optimization
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('ANEMOI_INFERENCE_NUM_CHUNKS', '16')
    
    # Create forecast runner
    runner = ForecastRunner(checkpoint, device=device)
    
    # Get latest available date
    date = OpendataClient("ecmwf").latest()
    
    # Prepare input state (control run)
    input_state = runner.prepare_input_state(date=date, ensemble_number=None)
    
    # Create output directory
    output_dir = "forecast_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    date_str = date.strftime("%Y%m%d_%H%M")
    grib_file = f"{output_dir}/aifs_ens_forecast_{date_str}.grib"
    netcdf_file = f"{output_dir}/aifs_ens_forecast_{date_str}.nc"
    
    # Run forecast with outputs
    forecast_states = runner.run_forecast(
        input_state=input_state,
        lead_time=lead_time,
        grib_output=grib_file,
        netcdf_output=netcdf_file,
        print_progress=True
    )
    
    print(f"\nForecast completed successfully!")
    print(f"Output files:")
    print(f"  GRIB: {grib_file}")
    print(f"  NetCDF: {netcdf_file}")
    
    return forecast_states


if __name__ == "__main__":
    forecast_states = main()