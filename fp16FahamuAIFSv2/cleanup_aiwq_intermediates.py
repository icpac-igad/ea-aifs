#!/usr/bin/env python3
"""
Cleanup script for AIWQ submission intermediates.

Removes large temporary files after successful zip creation:
- ensemble_nc_files/  (50 members × ~130 MB = ~6.5 GB)
- aiwq_individual_<date>/  (6 individual .nc files after zipping)
- ensemble_icechunk_store/ (optional, for fresh re-runs)

Usage:
    python cleanup_aiwq_intermediates.py --date 20260702 --keep-store
    python cleanup_aiwq_intermediates.py --all
"""

import argparse
import shutil
import os
from pathlib import Path
from glob import glob

def cleanup(date: str = None, all_mode: bool = False, keep_store: bool = False, dry_run: bool = False):
    """Remove intermediate files from AIWQ submission."""

    removed_files = []
    removed_dirs = []
    total_size_mb = 0

    print("🧹 AIWQ Intermediates Cleanup")
    print("=" * 60)

    # 1. Cleanup ensemble NetCDF files
    if os.path.exists("ensemble_nc_files"):
        size_mb = sum(f.stat().st_size for f in Path("ensemble_nc_files").glob("**/*") if f.is_file()) / 1024 / 1024
        total_size_mb += size_mb
        if dry_run:
            print(f"  [DRY RUN] Would remove: ensemble_nc_files/ ({size_mb:.1f} MB)")
        else:
            print(f"  🗑️  Removing ensemble_nc_files/ ({size_mb:.1f} MB)...", end="", flush=True)
            shutil.rmtree("ensemble_nc_files")
            print(" ✅")
        removed_dirs.append("ensemble_nc_files")

    # 2. Cleanup individual submission files directory
    if date:
        individual_dir = f"aiwq_individual_{date}"
        if os.path.exists(individual_dir):
            size_mb = sum(f.stat().st_size for f in Path(individual_dir).glob("**/*") if f.is_file()) / 1024 / 1024
            total_size_mb += size_mb
            if dry_run:
                print(f"  [DRY RUN] Would remove: {individual_dir}/ ({size_mb:.1f} MB)")
            else:
                print(f"  🗑️  Removing {individual_dir}/ ({size_mb:.1f} MB)...", end="", flush=True)
                shutil.rmtree(individual_dir)
                print(" ✅")
            removed_dirs.append(individual_dir)

    # 3. Cleanup icechunk store (optional)
    if all_mode or (date and not keep_store):
        if os.path.exists("ensemble_icechunk_store"):
            size_mb = sum(f.stat().st_size for f in Path("ensemble_icechunk_store").glob("**/*") if f.is_file()) / 1024 / 1024
            total_size_mb += size_mb
            if dry_run:
                print(f"  [DRY RUN] Would remove: ensemble_icechunk_store/ ({size_mb:.1f} MB)")
            else:
                print(f"  🗑️  Removing ensemble_icechunk_store/ ({size_mb:.1f} MB)...", end="", flush=True)
                shutil.rmtree("ensemble_icechunk_store")
                print(" ✅")
            removed_dirs.append("ensemble_icechunk_store")

    # 4. Find and cleanup old climatology files (optional)
    if all_mode and date:
        # Keep only the current date's climatology files
        for clim_file in glob("*quintile*.nc"):
            file_date = clim_file.split("_")[-1].split(".")[0]
            if file_date != date:
                size_mb = Path(clim_file).stat().st_size / 1024 / 1024
                total_size_mb += size_mb
                if dry_run:
                    print(f"  [DRY RUN] Would remove: {clim_file} ({size_mb:.1f} MB)")
                else:
                    print(f"  🗑️  Removing old climatology: {clim_file} ({size_mb:.1f} MB)...", end="", flush=True)
                    os.remove(clim_file)
                    print(" ✅")
                removed_files.append(clim_file)

    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Directories removed: {len(removed_dirs)}")
    print(f"   Files removed: {len(removed_files)}")
    print(f"   Space freed: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")

    if dry_run:
        print(f"\n💡 Use --run to actually delete these files")
    else:
        print(f"\n✅ Cleanup complete!")

    return total_size_mb

def main():
    parser = argparse.ArgumentParser(
        description="Cleanup AIWQ submission intermediate files"
    )
    parser.add_argument('--date', default=None,
                       help='Forecast date (YYYYMMDD) to cleanup individual files for')
    parser.add_argument('--all', action='store_true', default=False,
                       help='Full cleanup: ensemble_nc_files + aiwq_individual + icechunk_store')
    parser.add_argument('--keep-store', action='store_true', default=False,
                       help='Keep icechunk store for future re-runs (default: remove if --all)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                       help='Show what would be deleted without actually deleting')

    args = parser.parse_args()

    cleanup(
        date=args.date,
        all_mode=args.all,
        keep_store=args.keep_store,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
