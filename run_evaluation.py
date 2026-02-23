"""
Main script to run the evaluation of LUQ calculation.
arosquete - 2026/02/13
"""

# Imports
import argparse
import os
from pathlib import Path

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import numpy as np
import pandas as pd

from config import DATASETS, DRIFTER_COLUMNS, DRIFTERS_PATH, EVAL_PARAMS, OUTPUT_PATH
from utils import (
    L_characteristic_km,
    filter_by_temporal_range,
    load_dataset_files,
    load_drifter_file,
    luq_case,
    prep_dataset,
    prep_drifter,
    save_results_csv,
    validate_temporal_coverage,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run LUQ evaluation for FRESH-CARE surface currents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=None,
        help="Datasets to evaluate (default: all configured datasets)",
    )

    # Temporal parameters
    parser.add_argument(
        "--tau_days",
        nargs="+",
        type=int,
        default=None,
        help=f"Integration times in days (default: {EVAL_PARAMS['tau_days']})",
    )

    parser.add_argument(
        "--r_kms",
        nargs="+",
        type=int,
        default=None,
        help=f"Neighborhood radii in km (default: {EVAL_PARAMS['r_kms']})",
    )

    # Temporal range filtering
    parser.add_argument(
        "--eval_start",
        type=str,
        default=None,
        help="Start date for evaluation (YYYY-MM-DD, default: use all available)",
    )

    parser.add_argument(
        "--eval_end",
        type=str,
        default=None,
        help="End date for evaluation (YYYY-MM-DD, default: use all available)",
    )

    # File selection
    parser.add_argument(
        "--drifter_files",
        nargs="+",
        default=None,
        help="Specific drifter files to process (default: all in directory)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory (default: {OUTPUT_PATH})",
    )

    # Technical parameters
    parser.add_argument(
        "--dt_sec",
        type=int,
        default=None,
        help=f"Integration time step in seconds (default: {EVAL_PARAMS['dt_sec']})",
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=None,
        help=f"Neighborhood grid points X (default: {EVAL_PARAMS['nx']})",
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=None,
        help=f"Neighborhood grid points Y (default: {EVAL_PARAMS['ny']})",
    )

    parser.add_argument(
        "--min_valid_frac",
        type=float,
        default=None,
        help=f"Minimum valid fraction for seeds (default: {EVAL_PARAMS['min_valid_frac']})",
    )

    # Utility flags
    parser.add_argument(
        "--list_datasets", action="store_true", help="List available datasets and exit"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be evaluated without running",
    )

    return parser.parse_args()


def run_evaluation(
    datasets_to_eval=None,
    tau_days=None,
    r_kms=None,
    drifter_files=None,
    output_dir=None,
    eval_start=None,
    eval_end=None,
    dry_run=False,
    **kwargs,
):
    """
    Run LUQ evaluation with configurable parameters

    Parameters:
    -----------
    datasets_to_eval : list, optional
        Which datasets to evaluate (default: all in config)
    tau_days : list, optional
        Integration times in days (default: from config)
    r_kms : list, optional
        Neighborhood radii in km (default: from config)
    drifter_files : list, optional
        Specific drifter files to process (default: all in directory)
    output_dir : str/Path, optional
        Output directory (default: from config)
    eval_start : str, optional
        Start date for evaluation "YYYY-MM-DD"
    eval_end : str, optional
        End date for evaluation "YYYY-MM-DD"
    dry_run : bool, optional
        If True, only show what would be evaluated
    **kwargs : dict
        Other parameters to override (dt_sec, nx, ny, min_valid_frac)

    Returns:
    --------
    dict : LUQ evaluation results (None if dry_run=True)
    """

    # Use config defaults or override
    if datasets_to_eval is None:
        datasets_to_eval = list(DATASETS.keys())
    if tau_days is None:
        tau_days = EVAL_PARAMS["tau_days"]
    if r_kms is None:
        r_kms = EVAL_PARAMS["r_kms"]
    if output_dir is None:
        output_dir = OUTPUT_PATH

    # Get other params
    dt_sec = kwargs.get("dt_sec", EVAL_PARAMS["dt_sec"])
    nx = kwargs.get("nx", EVAL_PARAMS["nx"])
    ny = kwargs.get("ny", EVAL_PARAMS["ny"])
    min_valid_frac = kwargs.get("min_valid_frac", EVAL_PARAMS["min_valid_frac"])

    # Convert tau to seconds and get max for validation
    tau_secs = [np.int64(d * 24 * 3600) for d in tau_days]
    tau_max_sec = max(tau_secs)

    # Print evaluation summary
    print("=== EVALUATION CONFIGURATION ===")
    print(f"Datasets: {datasets_to_eval}")
    print(f"tau_days: {tau_days}")
    print(f"r_kms: {r_kms}")
    print(f"Integration step: {dt_sec}s, Grid: {nx}x{ny}, Min valid: {min_valid_frac}")
    if eval_start or eval_end:
        print(f"Temporal range: {eval_start or 'start'} to {eval_end or 'end'}")
    if drifter_files:
        print(f"Specific drifter files: {len(drifter_files)} files")
    print(f"Output directory: {output_dir}")

    if dry_run:
        print("\n*** DRY RUN - Would evaluate with above parameters ***")
        return None

    # 1) Load and validate datasets
    print("\n=== Loading Datasets ===")
    datasets = {}
    for name in datasets_to_eval:
        if name not in DATASETS:
            print(f"Warning: Dataset {name} not found in config")
            continue

        print(f"Loading {name}...")
        try:
            ds = load_dataset_files(
                DATASETS[name]["path"],
                eval_start=eval_start,
                eval_end=eval_end,
                tau_max_sec=tau_max_sec,
            )
            datasets[name] = prep_dataset(
                ds,
                u_name=DATASETS[name]["u_name"],
                v_name=DATASETS[name]["v_name"],
                time_name=DATASETS[name]["time_name"],
                lat_name=DATASETS[name]["lat_name"],
                lon_name=DATASETS[name]["lon_name"],
            )
            print(f"  ✓ Loaded {name}: {len(datasets[name]['time_sec'])} time steps")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets loaded successfully")

    # 2) Load and validate drifters
    print("\n=== Loading Drifters ===")
    drifters_path = Path(DRIFTERS_PATH)

    if drifter_files is None:
        # Load all drifter files
        drifter_files = list(drifters_path.glob("*.csv")) + list(
            drifters_path.glob("*.nc")
        )
    else:
        # Convert to Path objects
        drifter_files = [drifters_path / f for f in drifter_files]

    print(f"Found {len(drifter_files)} drifter files")

    # Load and validate drifters
    valid_drifters = {}
    for i, drifter_file in enumerate(drifter_files):
        try:
            df = load_drifter_file(drifter_file)
            drifter_data = prep_drifter(
                df,
                time_col=DRIFTER_COLUMNS["time_col"],
                lon_col=DRIFTER_COLUMNS["lon_col"],
                lat_col=DRIFTER_COLUMNS["lat_col"],
            )

            # Apply temporal filtering if specified
            filtered_data = filter_by_temporal_range(drifter_data, eval_start, eval_end)
            if filtered_data is None:
                print(
                    f"  ✗ Drifter {drifter_file.name} - no data in specified temporal range"
                )
                continue

            # Use filtered data for validation
            drifter_data = filtered_data

            # Validate temporal coverage against each dataset
            valid_for_datasets = {}
            for dataset_name, field in datasets.items():
                is_valid, msg = validate_temporal_coverage(
                    drifter_data, field, tau_max_sec
                )
                if is_valid:
                    valid_for_datasets[dataset_name] = True
                else:
                    print(
                        f"  Drifter {drifter_file.name} invalid for {dataset_name}: {msg}"
                    )

            if valid_for_datasets:
                valid_drifters[i] = {
                    "data": drifter_data,
                    "filename": drifter_file.name,
                    "valid_datasets": list(valid_for_datasets.keys()),
                }
                print(
                    f"  ✓ Drifter {i:03d}: {drifter_file.name} ({len(drifter_data['time_sec'])} points) - valid for {len(valid_for_datasets)} datasets"
                )
            else:
                print(f"  ✗ Drifter {drifter_file.name} - no valid datasets")

        except Exception as e:
            print(f"  ✗ Failed to load {drifter_file.name}: {e}")
            continue

    if not valid_drifters:
        raise ValueError("No valid drifters found")

    print(f"Proceeding with {len(valid_drifters)} valid drifters")

    # 3) Run LUQ evaluation
    print("\n=== Running LUQ Evaluation ===")
    LUQ_results = {}

    for drifter_idx, drifter_info in valid_drifters.items():
        print(f"Processing drifter {drifter_idx:03d}: {drifter_info['filename']}")

        drifter = drifter_info["data"]
        LUQ_results[drifter_idx] = {
            "filename": drifter_info["filename"],
            "L_characteristic_km": L_characteristic_km(
                drifter["lon_rad"], drifter["lat_rad"]
            ),
        }

        for dataset_name in drifter_info["valid_datasets"]:
            field = datasets[dataset_name]
            LUQ_results[drifter_idx][dataset_name] = {}

            for r_km in r_kms:
                LUQ_results[drifter_idx][dataset_name][r_km] = {}

                for tau_sec in tau_secs:
                    luq_values = []
                    valid_fracs = []
                    individual_calculations = []  # Store individual calculations
                    n_cases = 0

                    # Loop through all drifter positions
                    for i in range(len(drifter["time_sec"])):
                        t0_sec = drifter["time_sec"][i]
                        t1_sec = t0_sec + tau_sec

                        # Check if final time is within dataset range
                        if t1_sec > field["time_sec"][-1]:
                            continue

                        # Check if drifter has data at t1
                        idx1_candidates = np.where(drifter["time_sec"] == t1_sec)[0]
                        if len(idx1_candidates) == 0:
                            continue
                        idx1 = idx1_candidates[0]

                        # Get positions
                        lon0_rad = drifter["lon_rad"][i]
                        lat0_rad = drifter["lat_rad"][i]
                        lon_target_rad = drifter["lon_rad"][idx1]
                        lat_target_rad = drifter["lat_rad"][idx1]

                        # Compute LUQ for this case
                        luq_mean_km, valid_frac = luq_case(
                            field,
                            t0_sec,
                            tau_sec,
                            dt_sec,
                            lon0_rad,
                            lat0_rad,
                            lon_target_rad,
                            lat_target_rad,
                            r_km,
                            nx=nx,
                            ny=ny,
                            min_valid_frac=min_valid_frac,
                        )

                        if not np.isnan(luq_mean_km):
                            luq_values.append(luq_mean_km)
                            valid_fracs.append(valid_frac)
                            n_cases += 1
                            
                            # Store individual calculation for CSV
                            L_char = LUQ_results[drifter_idx]["L_characteristic_km"]
                            luq_normalized = luq_mean_km / L_char if L_char > 0 else np.nan
                            
                            # Convert time to datetime for CSV
                            t0_datetime = pd.to_datetime(
                                t0_sec, origin='1950-01-01', unit='s'
                            )  # Proper conversion from 1950 epoch
                            
                            individual_calculations.append({
                                "t0_datetime": t0_datetime,
                                "luq_km": luq_mean_km,
                                "luq_normalized": luq_normalized,
                                "valid_frac": valid_frac
                            })

                    # Aggregate results for this drifter-dataset-r_km-tau combination
                    if n_cases > 0:
                        luq_mean_over_cases = np.mean(luq_values).astype(np.float32)
                        valid_frac_over_cases = np.mean(valid_fracs).astype(np.float32)
                        L_char = LUQ_results[drifter_idx]["L_characteristic_km"]
                        luq_normalized = (
                            luq_mean_over_cases / L_char if L_char > 0 else np.nan
                        )
                    else:
                        luq_mean_over_cases = np.float32(np.nan)
                        valid_frac_over_cases = np.float32(0.0)
                        luq_normalized = np.nan

                    LUQ_results[drifter_idx][dataset_name][r_km][tau_sec] = {
                        "luq_mean_km": luq_mean_over_cases,
                        "luq_normalized": luq_normalized,
                        "valid_frac": valid_frac_over_cases,
                        "n_cases": n_cases,
                        "individual_calculations": individual_calculations,  # Add individual data
                    }

    # 4) Save results
    print("\n=== Saving Results ===")
    save_results_csv(LUQ_results, output_dir, tau_days, r_kms)

    return LUQ_results


if __name__ == "__main__":
    args = parse_arguments()

    # Handle utility commands
    if args.list_datasets:
        print("Available datasets:")
        for name, config in DATASETS.items():
            print(f"  {name}: {config['path']}")
        exit(0)

    # Convert args to function parameters
    kwargs = {}
    for param in ["dt_sec", "nx", "ny", "min_valid_frac"]:
        value = getattr(args, param, None)
        if value is not None:
            kwargs[param] = value

    # Run evaluation
    results = run_evaluation(
        datasets_to_eval=args.datasets,
        tau_days=args.tau_days,
        r_kms=args.r_kms,
        drifter_files=args.drifter_files,
        output_dir=args.output_dir,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        dry_run=args.dry_run,
        **kwargs,
    )

    if not args.dry_run:
        print(
            f"\nEvaluation completed! Results saved to {args.output_dir or OUTPUT_PATH}"
        )
