#!/bin/bash

set -e

anaconda_dir=<Your Anaconda Dir> # '/vol/research/yc/miniconda3'

. $anaconda_dir'/etc/profile.d/conda.sh'
conda remove -n ein --all -y
conda create -n ein python=3.7 -y
conda activate ein

conda install -c anaconda pandas h5py ipython pyyaml pylint -y
conda install pytorch torchvision cudatoolkit=10.2 torchaudio -c pytorch -y
conda install -c conda-forge librosa pudb tqdm ruamel.yaml -y
conda install -c omnia termcolor -y
pip install tensorboard