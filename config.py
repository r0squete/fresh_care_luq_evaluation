"""
Configuration file for LUQ calculation.
arosquete - 2026/02/13
"""

# Imports
import numpy as np

# Physical constants
R_EARTH = 6371000.0  # Earth's radius in meters
LAT_MAX = np.float32(1.569051)  # ~89.9° in radians
EPOCH = np.datetime64("1950-01-01T00:00:00", "s")

# Evaluation parameters
EVAL_PARAMS = {
    "tau_days": [5, 10, 15],
    "r_kms": [5, 10, 15],
    "dt_sec": 3600,
    "nx": 100,
    "ny": 100,
    "min_valid_frac": 0.9,
}

# Dataset paths and configurations
DATASETS = {
    "ADT-SST": {
        "path": "/home/rosquete/Documents/FRESH-CARE/data/fusion_evaluation/ana_evaluation/currents/ADT-SST",
        "u_name": "ug",
        "v_name": "vg",
        "time_name": "time",
        "lat_name": "lat",
        "lon_name": "lon",
    },
    "ADT-SSS": {
        "path": "/home/rosquete/Documents/FRESH-CARE/data/fusion_evaluation/ana_evaluation/currents/ADT-SSS",
        "u_name": "ug",
        "v_name": "vg",
        "time_name": "time",
        "lat_name": "lat",
        "lon_name": "lon",
    },
    "ADT-0.05": {
        "path": "/home/rosquete/Documents/FRESH-CARE/data/fusion_evaluation/ana_evaluation/currents/ADT-0.05",
        "u_name": "ugos",
        "v_name": "vgos",
        "time_name": "time",
        "lat_name": "latitude",
        "lon_name": "longitude",
    },
    "ADT-0.25": {
        "path": "/home/rosquete/Documents/FRESH-CARE/data/fusion_evaluation/ana_evaluation/currents/ADT-0.05",
        "u_name": "ugos",
        "v_name": "vgos",
        "time_name": "time",
        "lat_name": "lat",
        "lon_name": "lon",
    },
    "OSCAR": {
        "path": "/home/rosquete/Documents/FRESH-CARE/data/other_datasets/OSCAR/data/selected_region",
        "u_name": "u",
        "v_name": "v",
        "time_name": "time",
        "lat_name": "lat",
        "lon_name": "lon",
    },
}


# Paths
DRIFTERS_PATH = "/home/rosquete/Documents/FRESH-CARE/data/fusion_evaluation/data_in_situ/global_drifters_program/selected_drifters/final_selection/"
OUTPUT_PATH = "../results/"


# Drifter column names
DRIFTER_COLUMNS = {"time_col": "time", "lon_col": "longitude", "lat_col": "latitude"}
