#!/bin/sh

if [ $# -ne 1 ] 
then
  echo "Usage generate_pointclouds.sh <dataset_dir>"
  exit 1
fi

dataset_dir=$1
files=$(find ${dataset_dir} -name "*.stl")
total_num_files=$(echo "$files" | wc -l)
current_file=0
IFS=$(echo -en "\n\b")

for file in $files
do
    let current_file=$((current_file+1))
    echo "Processing ${file} ($current_file / $total_num_files)"
    timeout 900 python generate_pointcloud.py "${file}" 1024 "${file}.npy"
done