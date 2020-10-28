#!/bin/bash

set -e

CONFIG_FILE='./configs/ein_seld/seld.yaml'

# Extract data
python seld/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_data' --dataset_type='dev'
python seld/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_data' --dataset_type='eval'

# Extract scalar
python seld/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_scalar' --num_workers=8

# Extract meta
python seld/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_meta' --dataset_type='dev'
python seld/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_meta' --dataset_type='eval'
