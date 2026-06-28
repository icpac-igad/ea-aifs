#!/usr/bin/env python3
"""
ERA5T Forecast Submission CLI - For ERA5T-based AI Weather Quest submissions
=============================================================================

Reads credentials from .env file and submits all forecasts for a given date.

Usage:
    python era5t_forecast_submission_cli.py --date 20260305
    python era5t_forecast_submission_cli.py --date 20260305 --output-dir ./results

Environment Variables (.env file):
    AIWQ_TEAM_NAME=YourTeamName
    AIWQ_MODEL_NAME_ERA5T=YourERA5TModelName
    AIWQ_MODEL_NAME=YourModelName  (fallback)
    AIWQ_PASSWORD=YourPassword
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import submission function from era5t_ensemble_quintile_analysis_cli
try:
    from era5t_ensemble_quintile_analysis_cli import submit_forecast
    SUBMIT_AVAILABLE = True
except ImportError:
    try:
        from ensemble_quintile_analysis_cli import submit_forecast
        SUBMIT_AVAILABLE = True
    except ImportError:
        SUBMIT_AVAILABLE = False
        print("Warning: submit_forecast not available. Install AI_WQ_package.")


def get_credentials(fp16: bool = False) -> Tuple[str, str, str]:
    """
    Load credentials from environment variables.

    Args:
        fp16: If True, try to use FP16-specific model name if available

    Returns:
        Tuple of (team_name, model_name, password)
    """
    team_name = os.getenv('AIWQ_TEAM_NAME')
    password = os.getenv('AIWQ_PASSWORD')

    # Use ERA5T-specific model name if available, then FP16, then default
    if fp16:
        model_name = (os.getenv('AIWQ_MODEL_NAME_ERA5T') or
                      os.getenv('AIWQ_MODEL_NAME_FP16') or
                      os.getenv('AIWQ_MODEL_NAME'))
    else:
        model_name = os.getenv('AIWQ_MODEL_NAME_ERA5T') or os.getenv('AIWQ_MODEL_NAME')

    # Validate credentials
    missing = []
    if not team_name:
        missing.append('AIWQ_TEAM_NAME')
    if not model_name:
        missing.append('AIWQ_MODEL_NAME')
    if not password:
        missing.append('AIWQ_PASSWORD')

    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}\n"
            f"Create a .env file with these variables or set them in your environment."
        )

    return team_name, model_name, password


def get_output_file(date_str: str, output_dir: str, fp16: bool) -> str:
    """
    Determine the quintile output file path.

    Args:
        date_str: Forecast date (YYYYMMDD)
        output_dir: Directory containing output files
        fp16: If True, look for FP16 output file

    Returns:
        Path to the quintile NC file
    """
    # Clean date format
    date_clean = date_str.split('_')[0] if '_' in date_str else date_str

    if fp16:
        filename = f'era5t_ensemble_quintile_probabilities_{date_clean}_fp16.nc'
    else:
        filename = f'era5t_ensemble_quintile_probabilities_{date_clean}.nc'

    return os.path.join(output_dir, filename)


def submit_all_forecasts(
    output_file: str,
    date_str: str,
    team_name: str,
    model_name: str,
    password: str,
    variables: Optional[List[str]] = None,
    weeks: Optional[List[str]] = None,
    dry_run: bool = False
) -> Tuple[int, int, List[str]]:
    """
    Submit all forecasts for the given variables and weeks.

    Args:
        output_file: Path to quintile probabilities NC file
        date_str: Forecast date (YYYYMMDD)
        team_name: AI Weather Quest team name
        model_name: Model name for submission
        password: Submission password
        variables: List of variables to submit (default: ['mslp', 'pr', 'tas'])
        weeks: List of week periods (default: ['1', '2'])
        dry_run: If True, don't actually submit

    Returns:
        Tuple of (successful_count, failed_count, failed_submissions)
    """
    if variables is None:
        variables = ['mslp', 'pr', 'tas']
    if weeks is None:
        weeks = ['1', '2']

    successful = 0
    failed = 0
    failed_submissions = []

    total_submissions = len(variables) * len(weeks)
    current = 0

    print(f"\nSubmitting {total_submissions} forecasts...")
    print(f"  Team: {team_name}")
    print(f"  Model: {model_name}")
    print(f"  Date: {date_str}")
    print(f"  File: {output_file}")
    print("-" * 50)

    for variable in variables:
        for week in weeks:
            current += 1
            submission_id = f"{variable}_week{week}"

            print(f"\n[{current}/{total_submissions}] Submitting {submission_id}...")

            if dry_run:
                print(f"  [DRY RUN] Would submit {variable} week {week}")
                successful += 1
                continue

            try:
                result = submit_forecast(
                    output_file,
                    variable,
                    date_str,
                    week,
                    team_name,
                    model_name,
                    password
                )

                if result is not None:
                    print(f"  Successfully submitted {submission_id}")
                    successful += 1
                else:
                    print(f"  Warning: {submission_id} returned None")
                    failed += 1
                    failed_submissions.append(submission_id)

            except Exception as e:
                print(f"  Failed: {e}")
                failed += 1
                failed_submissions.append(f"{submission_id}: {str(e)}")

    return successful, failed, failed_submissions


def main():
    parser = argparse.ArgumentParser(
        description="ERA5T Forecast Submission CLI for AI Weather Quest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (.env file):
    AIWQ_TEAM_NAME       - Your team name
    AIWQ_MODEL_NAME_ERA5T - Model name for ERA5T submissions (optional)
    AIWQ_MODEL_NAME      - Fallback model name
    AIWQ_PASSWORD        - Submission password

Examples:
    # Submit ERA5T forecasts
    python era5t_forecast_submission_cli.py --date 20260305

    # Dry run (no actual submission)
    python era5t_forecast_submission_cli.py --date 20260305 --dry-run

    # Submit specific variables only
    python era5t_forecast_submission_cli.py --date 20260305 --variables mslp tas
        """
    )

    parser.add_argument('--date', required=True,
                       help='Forecast date (YYYYMMDD format)')
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use FP16 output file and model name (default: True for ERA5T)')
    parser.add_argument('--output-dir', default='./',
                       help='Directory containing quintile output files')
    parser.add_argument('--output-file', default=None,
                       help='Explicit path to quintile NC file (overrides auto-detection)')
    parser.add_argument('--variables', nargs='+', default=['mslp', 'pr', 'tas'],
                       choices=['mslp', 'pr', 'tas'],
                       help='Variables to submit (default: mslp pr tas)')
    parser.add_argument('--weeks', nargs='+', default=['1', '2'],
                       choices=['1', '2'],
                       help='Week periods to submit (default: 1 2)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be submitted without actually submitting')
    parser.add_argument('--env-file', default='.env',
                       help='Path to .env file (default: .env)')

    args = parser.parse_args()

    # Load .env file if specified
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        print(f"Loaded credentials from {args.env_file}")

    # Check if submission is available
    if not SUBMIT_AVAILABLE and not args.dry_run:
        print("ERROR: submit_forecast not available. Cannot proceed.")
        print("Install AI_WQ_package or use --dry-run to test.")
        return 1

    # Clean date format
    date_str = args.date.split('_')[0] if '_' in args.date else args.date
    mode_label = "FP16" if args.fp16 else "FP32"

    print("=" * 60)
    print(f"AI Weather Quest Forecast Submission ({mode_label})")
    print("=" * 60)

    # Get credentials
    try:
        team_name, model_name, password = get_credentials(args.fp16)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = get_output_file(date_str, args.output_dir, args.fp16)

    # Check if output file exists
    if not os.path.exists(output_file):
        print(f"ERROR: Output file not found: {output_file}")
        print(f"\nRun era5t_ensemble_quintile_analysis_cli.py first:")
        print(f"  python era5t_ensemble_quintile_analysis_cli.py --date {date_str} --members 0-9")
        return 1

    # Submit forecasts
    successful, failed, failed_submissions = submit_all_forecasts(
        output_file=output_file,
        date_str=date_str,
        team_name=team_name,
        model_name=model_name,
        password=password,
        variables=args.variables,
        weeks=args.weeks,
        dry_run=args.dry_run
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"Mode: {mode_label}")
    print(f"Date: {date_str}")
    print(f"Team: {team_name}")
    print(f"Model: {model_name}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed_submissions:
        print(f"\nFailed submissions:")
        for item in failed_submissions:
            print(f"  - {item}")

    if args.dry_run:
        print("\n[DRY RUN] No actual submissions were made.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
