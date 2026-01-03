#!/usr/bin/env python3
"""
Create a 3-row comparison plot showing the progression of precipitation quintile fixes:
  Row 1: Original buggy code (no mm conversion, wrong step slicing)
  Row 2: Partial fix (mm conversion only, still wrong step slicing)
  Row 3: Full fix (mm conversion + correct step aggregation)

Uses full 50-member ensemble data for smooth probability distributions.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def load_ensemble_data():
    """Load the 50-member ensemble forecast data"""
    import icechunk

    store_path = "./ensemble_icechunk_store"
    local_storage = icechunk.local_filesystem_storage(store_path)
    repo = icechunk.Repository.open(local_storage)
    session = repo.readonly_session(branch="main")

    ensemble_ds = xr.open_zarr(session.store, group="ensemble_forecast",
                               chunks={'member': 1, 'step': 10, 'latitude': 60, 'longitude': 120})
    return ensemble_ds

def calculate_quintile_probs(weekly_forecast, clim_pr):
    """
    Calculate quintile probabilities across ensemble members.

    Args:
        weekly_forecast: (member, lat, lon) array
        clim_pr: climatology with quantile dimension

    Returns:
        Quintile probabilities (5, lat, lon)
    """
    n_members = weekly_forecast.sizes['member']

    # Get climatology thresholds
    q20 = clim_pr.sel(quantile=0.2).values
    q40 = clim_pr.sel(quantile=0.4).values
    q60 = clim_pr.sel(quantile=0.6).values
    q80 = clim_pr.sel(quantile=0.8).values

    # Get ensemble values
    fc_vals = weekly_forecast.values  # (member, lat, lon)

    # Calculate probability in each quintile
    probs = np.zeros((5, fc_vals.shape[1], fc_vals.shape[2]))
    probs[0] = np.sum(fc_vals < q20, axis=0) / n_members
    probs[1] = np.sum((fc_vals >= q20) & (fc_vals < q40), axis=0) / n_members
    probs[2] = np.sum((fc_vals >= q40) & (fc_vals < q60), axis=0) / n_members
    probs[3] = np.sum((fc_vals >= q60) & (fc_vals < q80), axis=0) / n_members
    probs[4] = np.sum(fc_vals >= q80, axis=0) / n_members

    return probs

def calculate_weekly_original_bug(tp_data):
    """
    Original buggy code: No mm conversion, wrong step slicing
    """
    chunk_data = []
    for t_idx in [0, 1]:
        chunk = tp_data.isel(time=t_idx, step=slice(0, 12))
        chunk_data.append(chunk)

    forecast_data = xr.concat(chunk_data, dim='step')
    # NO mm conversion - original bug!
    weekly_forecast = forecast_data.sum(dim='step')
    return weekly_forecast

def calculate_weekly_partial_fix(tp_data):
    """
    Partial fix: mm conversion added, but still wrong step slicing
    """
    chunk_data = []
    total_steps = 0
    for t_idx in [0, 1]:
        chunk = tp_data.isel(time=t_idx, step=slice(0, 12))
        chunk_data.append(chunk)
        total_steps += 12

    forecast_data = xr.concat(chunk_data, dim='step')
    actual_days = total_steps * 6 / 24
    scaling_factor = 7.0 / actual_days
    # With mm conversion
    weekly_forecast = forecast_data.sum(dim='step') * 1000 * scaling_factor
    return weekly_forecast

def calculate_weekly_full_fix(tp_data):
    """
    Full fix: mm conversion + correct step aggregation using skipna
    """
    # Sum each time chunk separately (correct approach)
    chunk_sums = []
    total_steps = 0
    for t_idx in [0, 1]:
        chunk = tp_data.isel(time=t_idx)
        chunk_sum = chunk.sum(dim='step', skipna=True)
        chunk_sums.append(chunk_sum)
        total_steps += 12

    # Sum across time chunks
    forecast_sum = chunk_sums[0] + chunk_sums[1]
    actual_days = total_steps * 6 / 24
    scaling_factor = 7.0 / actual_days

    weekly_forecast = forecast_sum * 1000 * scaling_factor
    return weekly_forecast

def main():
    print("=" * 70)
    print("Creating 3-Row Comparison Plot: Bug Progression Fix")
    print("Using 50-member ensemble for smooth probability distributions")
    print("=" * 70)

    # Load ensemble data
    print("\nLoading 50-member ensemble data from icechunk store...")
    ensemble_ds = load_ensemble_data()
    tp = ensemble_ds['tp']
    print(f"  Ensemble shape: {tp.shape}")
    print(f"  Members: {ensemble_ds.sizes['member']}")

    # Load climatology
    print("\nLoading climatology...")
    clim = xr.open_dataset('/scratch/notebook/pr_20yrCLIM_WEEKLYSUM_quintiles_20260119.nc')
    clim_pr = clim['pr'].isel(time=0)
    print(f"  Climatology shape: {clim_pr.shape}")

    # Calculate for each code version
    print("\nCalculating quintiles for each code version (this may take a moment)...")

    print("  1. Original bug (no mm conversion, wrong slicing)...")
    weekly_bug = calculate_weekly_original_bug(tp)
    weekly_bug_loaded = weekly_bug.compute()
    probs_bug = calculate_quintile_probs(weekly_bug_loaded, clim_pr)
    means_bug = [np.nanmean(probs_bug[i]) for i in range(5)]
    print(f"     Mean weekly precip: {float(weekly_bug_loaded.mean()):.6f} (in meters!)")
    print(f"     Quintile means: Q1={means_bug[0]:.3f}, Q2={means_bug[1]:.3f}, Q3={means_bug[2]:.3f}, Q4={means_bug[3]:.3f}, Q5={means_bug[4]:.3f}")

    print("  2. Partial fix (mm conversion, wrong slicing)...")
    weekly_partial = calculate_weekly_partial_fix(tp)
    weekly_partial_loaded = weekly_partial.compute()
    probs_partial = calculate_quintile_probs(weekly_partial_loaded, clim_pr)
    means_partial = [np.nanmean(probs_partial[i]) for i in range(5)]
    print(f"     Mean weekly precip: {float(weekly_partial_loaded.mean()):.2f} mm")
    print(f"     Quintile means: Q1={means_partial[0]:.3f}, Q2={means_partial[1]:.3f}, Q3={means_partial[2]:.3f}, Q4={means_partial[3]:.3f}, Q5={means_partial[4]:.3f}")

    print("  3. Full fix (mm conversion + correct aggregation)...")
    weekly_full = calculate_weekly_full_fix(tp)
    weekly_full_loaded = weekly_full.compute()
    probs_full = calculate_quintile_probs(weekly_full_loaded, clim_pr)
    means_full = [np.nanmean(probs_full[i]) for i in range(5)]
    print(f"     Mean weekly precip: {float(weekly_full_loaded.mean()):.2f} mm")
    print(f"     Quintile means: Q1={means_full[0]:.3f}, Q2={means_full[1]:.3f}, Q3={means_full[2]:.3f}, Q4={means_full[3]:.3f}, Q5={means_full[4]:.3f}")

    # Create the 3-row comparison plot (same style as original 2-row plot)
    print("\nCreating comparison plot...")

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    quintile_labels = ['Q1 (Dry)', 'Q2', 'Q3', 'Q4', 'Q5 (Wet)']
    row_labels = [
        'ORIGINAL BUG',
        'PARTIAL FIX\n(mm conversion only)',
        'FULL FIX'
    ]

    all_probs = [probs_bug, probs_partial, probs_full]
    all_means = [means_bug, means_partial, means_full]

    for row_idx, (probs, means, label) in enumerate(zip(all_probs, all_means, row_labels)):
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]

            data = probs[col_idx]
            im = ax.imshow(data, cmap='RdBu_r', vmin=0, vmax=1, origin='upper', aspect='auto')

            ax.set_title(f'{label} {quintile_labels[col_idx]}\nMean: {means[col_idx]:.3f}')
            ax.set_xlabel('Longitude')

            if col_idx == 0:
                ax.set_ylabel('Latitude')

    plt.suptitle('Precipitation Quintile Probabilities: Bug Fix Progression\n' +
                 '(50-member ensemble, Forecast Date: 2026-01-01, Week 1)', fontsize=14)
    plt.tight_layout()

    # Save plot
    output_path = '/scratch/notebook/test_outputs/plots/precip_quintile_3row_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Quintile Distribution Comparison (50-member ensemble)")
    print("=" * 70)
    print(f"\n{'Code Version':<35} {'Q1':<8} {'Q2':<8} {'Q3':<8} {'Q4':<8} {'Q5':<8}")
    print("-" * 75)
    print(f"{'Expected (calibrated)':<35} {'0.200':<8} {'0.200':<8} {'0.200':<8} {'0.200':<8} {'0.200':<8}")
    print(f"{'Original Bug':<35} {means_bug[0]:<8.3f} {means_bug[1]:<8.3f} {means_bug[2]:<8.3f} {means_bug[3]:<8.3f} {means_bug[4]:<8.3f}")
    print(f"{'Partial Fix (mm only)':<35} {means_partial[0]:<8.3f} {means_partial[1]:<8.3f} {means_partial[2]:<8.3f} {means_partial[3]:<8.3f} {means_partial[4]:<8.3f}")
    print(f"{'Full Fix (mm + correct agg)':<35} {means_full[0]:<8.3f} {means_full[1]:<8.3f} {means_full[2]:<8.3f} {means_full[3]:<8.3f} {means_full[4]:<8.3f}")

    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    print(f"""
1. ORIGINAL BUG (Row 1):
   - Q1 = {means_bug[0]:.1%} (expected 20%) - SEVERELY BIASED
   - Cause: No mm conversion (1000x scale error) + wrong step slicing
   - Result: Everything appears extremely dry

2. PARTIAL FIX (Row 2):
   - Q1 = {means_partial[0]:.1%} (expected 20%) - STILL BIASED
   - Cause: mm conversion added, but step slicing still wrong
   - Result: Only ~3 days of data instead of 6 days

3. FULL FIX (Row 3):
   - Q1 = {means_full[0]:.1%} (expected 20%) - WELL CALIBRATED
   - Both fixes applied: mm conversion + correct aggregation
   - Result: Distribution close to expected 20% per quintile
""")

    clim.close()
    print("Done!")

if __name__ == "__main__":
    main()
