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
    to_epoch_seconds_datetime64,
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
        default=list(DATASETS.keys()),
        help="Datasets to evaluate",
    )

    # Temporal parameters
    parser.add_argument(
        "--tau_days",
        nargs="+",
        type=int,
        default=EVAL_PARAMS["tau_days"],
        help="Integration times in days",
    )

    # Spatial parameters
    parser.add_argument(
        "--r_kms",
        nargs="+",
        type=int,
        default=EVAL_PARAMS["r_kms"],
        help="Neighborhood radii in km",
    )

    # Temporal range filtering
    parser.add_argument(
        "--eval_start",
        type=str,
        required=True,
        help="Start date for evaluation (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--eval_end",
        type=str,
        required=True,
        help="End date for evaluation (YYYY-MM-DD)",
    )

    # File selection
    parser.add_argument(
        "--drifter_files",
        nargs="+",
        help="Specific drifter files in <directory> to process, (default: all .csv in <drifters directory)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_PATH,
        help="Output directory",
    )

    # Technical parameters
    parser.add_argument(
        "--dt_sec",
        type=int,
        default=EVAL_PARAMS["dt_sec"],
        help="Integration time step in seconds",
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=EVAL_PARAMS["nx"],
        help="Neighborhood grid points X",
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=EVAL_PARAMS["ny"],
        help="Neighborhood grid points Y",
    )

    parser.add_argument(
        "--min_valid_frac",
        type=float,
        default=EVAL_PARAMS["min_valid_frac"],
        help="Minimum valid fraction for seeds",
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
    eval_start : str, required
        Start date for evaluation "YYYY-MM-DD"
    eval_end : str, required
        End date for evaluation "YYYY-MM-DD"
    dry_run : bool, optional
        If True, only show what would be evaluated
    **kwargs : dict
        Other parameters to override (dt_sec, nx, ny, min_valid_frac)

    Returns:
    --------
    dict : LUQ evaluation results (None if dry_run=True)
    """
    if eval_start is None or eval_end is None:
        raise ValueError("eval_start and eval_end are required")

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
    print(f"Temporal range: {eval_start} to {eval_end}")
    if drifter_files:
        print(f"Specific drifter files: {len(drifter_files)} files")
    print(f"Output directory: {output_dir}")

    if dry_run:
        print("\n*** DRY RUN - Would evaluate with above parameters ***")
        return None

    # 1) Load and validate datasets
    datasets = {}
    for name in datasets_to_eval:
        if name not in DATASETS:
            print(f"Warning: Dataset {name} not found in config")
            continue
        try:
            print(f"Loading dataset: {name}")
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

        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
            continue
    if not datasets:
        raise ValueError("No datasets loaded successfully")

    # 2) Load and validate drifters
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

            # Apply temporal filtering to eval window
            filtered_data = filter_by_temporal_range(
                drifter_data, eval_start, eval_end, tau_max_sec
            )
            if filtered_data is None:
                print(
                    f"  ✗ Drifter {drifter_file.name} - no data in specified temporal range"
                )
                continue
            print(
                f"  ✓ Drifter {drifter_file.name} - {len(filtered_data['time_sec'])} records in specified temporal range"
            )
            drifter_data = filtered_data

            # Clip drifter to each field's temporal range and validate coverage
            valid_for_datasets = {}
            for dataset_name, field in datasets.items():
                drifter_for_field = filter_by_temporal_range(
                    drifter_data,
                    eval_start,
                    eval_end,
                    tau_max_sec,
                    field_start_sec=field["time_sec"][0],
                    field_end_sec=field["time_sec"][-1],
                )
                if drifter_for_field is None:
                    print(
                        f"  ✗ Drifter {drifter_file.name} - insufficient coverage for '{dataset_name}'"
                    )
                    continue

                valid_for_datasets[dataset_name] = drifter_for_field

            if valid_for_datasets:
                valid_drifters[i] = {
                    "data": drifter_data,
                    "filename": drifter_file.name,
                    "valid_datasets": list(valid_for_datasets.keys()),
                    "clipped_data": valid_for_datasets,
                }

        except Exception as e:
            print(f"  ✗ Failed to load {drifter_file.name}: {e}")
            continue
    print(f"Proceeding with {len(valid_drifters)} valid drifters")
    if not valid_drifters:
        raise ValueError("No valid drifters found")
    print("Running LUQ Evaluation...")

    # Pre-compute t0 iteration bounds (same for all drifters/datasets/tau)
    t0_start_sec = int(
        to_epoch_seconds_datetime64(
            np.array([pd.Timestamp(eval_start).to_datetime64().astype("datetime64[s]")])
        )[0]
    )
    t0_end_sec = int(
        to_epoch_seconds_datetime64(
            np.array([pd.Timestamp(eval_end).to_datetime64().astype("datetime64[s]")])
        )[0]
    )

    LUQ_results = {}
    for drifter_idx, drifter_info in valid_drifters.items():
        print(f"[EVAL] Drifter {drifter_idx:03d}: {drifter_info['filename']}")
        # L_char is a single spatial scale for the drifter, computed on the
        # eval-window-clipped trajectory.
        L_char = L_characteristic_km(
            drifter_info["data"]["lon_rad"], drifter_info["data"]["lat_rad"]
        )
        LUQ_results[drifter_idx] = {
            "filename": drifter_info["filename"],
        }
        for dataset_name in drifter_info["valid_datasets"]:
            field = datasets[dataset_name]
            drifter = drifter_info["clipped_data"][dataset_name]
            LUQ_results[drifter_idx][dataset_name] = {}
            for r_km in r_kms:
                LUQ_results[drifter_idx][dataset_name][r_km] = {}
                for tau_sec in tau_secs:
                    luq_values = []
                    valid_fracs = []
                    individual_calculations = []
                    n_cases = 0

                    # Iterate t0 over [eval_start, eval_end] in 1-day steps.
                    for t0_sec in range(t0_start_sec, t0_end_sec + 1, 86400):
                        t0 = np.int64(t0_sec)
                        t1_sec = t0 + tau_sec
                        idx0_candidates = np.where(drifter["time_sec"] == t0)[0]
                        if len(idx0_candidates) == 0:
                            continue
                        if t1_sec > field["time_sec"][-1]:
                            continue
                        idx1_candidates = np.where(drifter["time_sec"] == t1_sec)[0]
                        if len(idx1_candidates) == 0:
                            continue
                        idx1 = idx1_candidates[0]
                        idx0 = idx0_candidates[0]
                        lon0_rad = drifter["lon_rad"][idx0]
                        lat0_rad = drifter["lat_rad"][idx0]
                        lon_target_rad = drifter["lon_rad"][idx1]
                        lat_target_rad = drifter["lat_rad"][idx1]
                        luq_mean_km, valid_frac = luq_case(
                            field,
                            t0,
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
                        t0_datetime = pd.Timestamp("1950-01-01") + pd.Timedelta(
                            seconds=int(t0)
                        )
                        luq_normalized = (
                            luq_mean_km / L_char
                            if (not np.isnan(luq_mean_km) and L_char > 0)
                            else np.nan
                        )
                        # Always record the timestep in the time series,
                        # even if valid_frac < min_valid_frac (luq_km = NaN).
                        individual_calculations.append(
                            {
                                "t0_datetime": t0_datetime,
                                "lon0_deg": float(np.rad2deg(lon0_rad)),
                                "lat0_deg": float(np.rad2deg(lat0_rad)),
                                "luq_km": luq_mean_km,
                                "luq_normalized": luq_normalized,
                                "valid_frac": valid_frac,
                            }
                        )
                        # Only valid cases contribute to the aggregate mean.
                        if not np.isnan(luq_mean_km):
                            luq_values.append(luq_mean_km)
                            valid_fracs.append(valid_frac)
                            n_cases += 1
                    # Aggregate results
                    if n_cases > 0:
                        luq_mean_over_cases = np.mean(luq_values).astype(np.float32)
                        valid_frac_over_cases = np.mean(valid_fracs).astype(np.float32)
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
                        "L_characteristic_km": L_char,
                        "individual_calculations": individual_calculations,
                    }
    print("LUQ evaluation finished.")
    print("\n=== Saving Results ===")
    save_results_csv(LUQ_results, output_dir, tau_days, r_kms, eval_start=eval_start)
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
    kwargs = {
        param: getattr(args, param)
        for param in ["dt_sec", "nx", "ny", "min_valid_frac"]
    }

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
