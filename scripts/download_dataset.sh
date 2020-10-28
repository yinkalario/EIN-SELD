#!/bin/bash

set -e

# check bin requirement
command -v wget >/dev/null 2>&1 || { echo 'wget is missing' >&2; exit 1; }
command -v zip >/dev/null 2>&1 || { echo 'zip is missing' >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo 'unzip is missing' >&2; exit 1; }

## dcase 2020 Task 3
Dataset_dir='_dataset'
Dataset_root=$Dataset_dir'/dataset_root'
mkdir -p $Dataset_root
Download_packages_dir=$Dataset_dir'/downloaded_packages'
mkdir -p $Download_packages_dir

# dev
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/foa_dev.z01'
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/foa_dev.z02'
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/foa_dev.zip'
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/foa_eval.zip'
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/metadata_dev.zip'
wget -P $Download_packages_dir 'https://zenodo.org/record/4064792/files/metadata_eval.zip'
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/mic_dev.z01'
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/mic_dev.z02'
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/mic_dev.zip'
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/mic_eval.zip'
wget -P $Download_packages_dir 'https://zenodo.org/record/3870859/files/README.md'

zip -s 0 $Download_packages_dir'/foa_dev.zip' --out $Download_packages_dir'/foa_dev_single.zip'
zip -s 0 $Download_packages_dir'/mic_dev.zip' --out $Download_packages_dir'/mic_dev_single.zip'

unzip $Download_packages_dir'/foa_dev_single.zip' -d $Dataset_root
unzip $Download_packages_dir'/mic_dev_single.zip' -d $Dataset_root
unzip $Download_packages_dir'/metadata_dev.zip' -d $Dataset_root
unzip $Download_packages_dir'/metadata_eval.zip' -d $Dataset_root
unzip $Download_packages_dir'/foa_eval.zip' -d $Dataset_root
unzip $Download_packages_dir'/mic_eval.zip' -d $Dataset_root