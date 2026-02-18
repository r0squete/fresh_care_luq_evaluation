# FRESH-CARE LUQ Evaluation

Lagrangian Uncertainty Quantification (LUQ) evaluation for FRESH-CARE surface current datasets.

## Description

This repository contains tools to evaluate the accuracy of surface current datasets by comparing predicted drifter trajectories with observed trajectories using the LUQ metric.

## Features

- Configurable evaluation parameters
- Multiple dataset support (ADT-SST, ADT-SSS, ADT-0.05...)
- Temporal filtering capabilities
- Command-line interface
- CSV output format

## Usage

### Basic evaluation
python run_evaluation.py

### Custom evaluation
python run_evaluation.py --datasets ADT-SST --tau_days 5 10 --eval_start 2021-10-01 --eval_end 2021-12-31 

### Available options 
python run_evaluation.py --help



## Configuration
Edit config.py to modify:
- Dataset paths and variable names
- Default evaluation parameters
- Output directories

## Requirements
- Python 3.8+
- NumPy
- Pandas
- xarray
- numba

## Installation
git clone https://github.com/yourusername/fresh-care-luq-evaluation.git
cd fresh-care-luq-evaluation
pip install -r requirements.txt

## Authors

- arosquete - 2026/02/13


