#!/bin/bash

#export ESOT500_DIR='/root/autodl-tmp/ESOT500'
#export ESOT500H_DIR='/root/autodl-tmp/ESOT500-H'
#export STARE_CKPTS_DIR='/root/autodl-tmp/STARE_trackers_more'


fps_options=(500 250 20)
window_options=(2 8 20 50)


# Run frame-based tracking
echo "Starting frame-based tracking tests with Stark_S baseline ..."
echo "------------------------------------"

# Loop through each dataset name
for fps in "${fps_options[@]}"; do
  for window in "${window_options[@]}"; do
    dataset="esoth_${fps}_${window}"
    echo "Running test for dataset: ${dataset}"
    python tracking/test.py hdetrack hdetrack_eventvot --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done