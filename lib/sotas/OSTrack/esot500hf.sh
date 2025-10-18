#!/bin/bash

fps_options=(500 250 20)
window_options=(2 8 20 50)

# Run frame-based tracking
echo "Starting frame-based tracking tests with OSTrack ..."
echo "------------------------------------"

# Loop through each dataset name
for fps in "${fps_options[@]}"; do
  for window in "${window_options[@]}"; do
    dataset="esoth_${fps}_${window}"
    echo "Running test for dataset: ${dataset}"
    python tracking/test.py ostrack esot500mix --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done