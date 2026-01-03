import xarray as xr
   import numpy as np
   import matplotlib.pyplot as plt

   # Load the output NetCDF file
   ds = xr.open_dataset('/scratch/notebook/ensemble_quintile_probabilities_20260101.nc')

   print("=== Ensemble Quintile Probabilities Output ===\n")
   print(f"Variables: {list(ds.data_vars)}")
   print(f"Dimensions: {dict(ds.sizes)}")
   print(f"Coordinates: {list(ds.coords)}")
   print(f"\nWeek values: {ds.week.values}")

   # Get precipitation quintiles
   tp_quintiles = ds['tp_quintiles']
   print(f"\nPrecipitation quintiles shape: {tp_quintiles.shape}")
   print(f"Precipitation quintiles dims: {tp_quintiles.dims}")

   # Create 2-row plot: Week 1 and Week 2
   fig, axes = plt.subplots(2, 5, figsize=(20, 8))

   quintile_labels = ['Q1 (Dry)', 'Q2', 'Q3', 'Q4', 'Q5 (Wet)']
   week_labels = ['Week 1', 'Week 2']

   for row_idx in range(2):
       # Get data for this week
       week_data = tp_quintiles.isel(time_week=row_idx)
       week_name = str(ds.week.values[row_idx])

       for col_idx in range(5):
           ax = axes[row_idx, col_idx]

           data = week_data.isel(quintile=col_idx)
           im = ax.imshow(data.values, cmap='RdBu_r', vmin=0, vmax=1, origin='upper', aspect='auto')

           mean_val = float(data.mean())
           ax.set_title(f'{week_labels[row_idx]} {quintile_labels[col_idx]}\nMean: {mean_val:.3f}')
           ax.set_xlabel('Longitude')

           if col_idx == 0:
               ax.set_ylabel('Latitude')

   # Add colorbar
   cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
   cbar = fig.colorbar(im, cax=cbar_ax)
   cbar.set_label('Probability', fontsize=12)

   plt.suptitle('Precipitation Quintile Probabilities (FIXED)\n' +
                'Forecast Date: 2026-01-01, 50-member Ensemble', fontsize=14, fontweight='bold')
   plt.tight_layout(rect=[0, 0, 0.9, 0.95])

   # Save plot
   output_path = '/scratch/notebook/test_outputs/plots/precip_quintile_week1_week2.png'
   plt.savefig(output_path, dpi=150, bbox_inches='tight')
   plt.close()

   print(f"\nPlot saved to: {output_path}")

   # Print summary statistics
   print("\n" + "=" * 70)
   print("SUMMARY: Precipitation Quintile Probabilities")
   print("=" * 70)

   for week_idx, week_name in enumerate(['Week 1', 'Week 2']):
       week_data = tp_quintiles.isel(time_week=week_idx)
       print(f"\n{week_name}:")
       print(f"  {'Quintile':<12} {'Mean Prob':<12} {'Expected':<12}")
       print(f"  {'-'*36}")
       for q_idx, q_label in enumerate(quintile_labels):
           mean_prob = float(week_data.isel(quintile=q_idx).mean())
           print(f"  {q_label:<12} {mean_prob:<12.3f} {'0.200':<12}")

   ds.close()
   print("\nDone!")
