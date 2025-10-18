#!/bin/bash

#python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
#python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# cp -r $STARE_CKPTS_DIR/sotas/stark_s lib/test/networks

setting_options=(s101 s107 s102 s103)

# Run stare tracking
echo "Starting stare tracking tests with Stark_S baseline ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options[@]}"; do
  echo "Running test for setting: ${setting}"
  python tracking/test_streaming.py hdetrack hdetrack_eventvot "${setting}" --dataset_name esot500hs
  echo "------------------------------------"
done

echo "All tracking tests completed for Stark."