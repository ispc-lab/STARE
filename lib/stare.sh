#!/bin/bash

#export ESOT500_DIR='/your/path/to/ESOT500'
#export STARE_CKPTS_DIR='/your/path/to/stare_ckpts'

conda activate stare

#############################################################
### Step 1: prepare the dataset
#############################################################

# Check if the ESOT500_DIR environment variable is set
if [ -z "$ESOT500_DIR" ]; then
  echo "Error: ESOT500_DIR environment variable is not set. Please define it before running."
  echo "For example: export ESOT500_DIR=\"/your/path/to/ESOT500\""
  exit 1
fi

ln -s $ESOT500_DIR data/ESOT500

# Define the optional values for FPS and WINDOW
fps_options=(500 250 20)
window_options=(2 50 100 150)

echo "Starting ESOT500 dataset preprocessing..."
echo "ESOT500 Data Directory: $ESOT500_DIR"
echo "------------------------------------"

# Loop through all FPS options
for fps in "${fps_options[@]}"; do
  # Loop through all WINDOW options
  for window in "${window_options[@]}"; do
    echo "Running with: fps=${fps}, window=${window}"
    python lib/event_utils_new/esot500_preprocess.py --path_to_data "$ESOT500_DIR" --fps "$fps" --window "$window"
    echo "------------------------------------"
  done
done

ln -s data/ESOT500/500_w2ms data/ESOT500/500

echo "All ESOT500 data preprocessing tasks completed."


#############################################################
### Step 2: test the trackers under PyTracking
#############################################################

# go to the pytracking directory
cd lib/pytracking

# Create the default local file for pytracking
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

# Copy the checkpoints to the pytracking networks directory
cp -r $STARE_CKPTS_DIR/pytracking lib/pytracking/pytracking/networks

# Run frame-based tracking
echo "Starting frame-based tracking tests for trackers under PyTracking..."
echo "------------------------------------"
python pytracking/run_experiment.py exp_frame esot500_frame_all

echo "------------------------------------"

# Run stare tracking
echo "Starting stare tracking tests for trackers under PyTracking..."
echo "------------------------------------"
python pytracking/run_experiment_streaming.py exp_stare esot500_stare_all

# align the prediction with GT timestamp
python eval/streaming_eval_v3.py exp_stare esot500_stare_all

echo "All tracking tests completed for trackers under PyTracking."

#############################################################
### Step 3: test the trackers under other frameworks
#############################################################

# go to the sotas directory
cd ../sotas

# Step 3.1: MixFormer

cd MixFormer

# Create the default local file for MixFormer
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Copy the checkpoints to the sotas networks directory
cp -r $STARE_CKPTS_DIR/sotas/mixformer_convmae_online lib/test/networks

# Run frame-based tracking
echo "Starting frame-based tracking tests with MixFormer ConvMAE Online baseline..."
echo "------------------------------------"

# Define the list of dataset names
dataset_names=(
    'esot_500_2' 'esot_250_2' 'esot_20_2'
    'esot_500_50' 'esot_250_50' 'esot_20_50'
    'esot_500_100' 'esot_250_100' 'esot_20_100'
    'esot_500_150' 'esot_250_150' 'esot_20_150'
)

# Loop through each dataset name
for dataset in "${dataset_names[@]}"; do
  echo "Running test for dataset: ${dataset}"
  python tracking/test.py mixformer_convmae_online baseline --dataset_name "${dataset}"
  echo "------------------------------------"
done

# Run stare tracking
echo "Starting stare tracking tests with MixFormer ConvMAE Online baseline..."
echo "------------------------------------"

# Define the list of setting options
setting_options=(s101 s102 s103 s104 s105 s106)

# Loop through each setting option
for setting in "${setting_options[@]}"; do
  echo "Running test for setting: ${setting}"
  python tracking/test_streaming.py mixformer_convmae_online baseline "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py mixformer_convmae_online baseline "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

echo "All tracking tests completed for MixFormer."

# Step 3.2: Stark

cd ../Stark

# Create the default local file for Stark
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Copy the checkpoints to the sotas networks directory
cp -r $STARE_CKPTS_DIR/sotas/stark_s lib/test/networks

# Run frame-based tracking
echo "Starting frame-based tracking tests with Stark_S baseline ..."
echo "------------------------------------"

# Loop through each dataset name
for dataset in "${dataset_names[@]}"; do
  echo "Running test for dataset: ${dataset}"
  python tracking/test.py stark_s baseline --dataset_name "${dataset}"
  echo "------------------------------------"
done

# Run stare tracking
echo "Starting stare tracking tests with Stark_S baseline ..."
echo "Dataset: esot500s"
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options[@]}"; do
  echo "Running test for setting: ${setting}"
  python tracking/test_streaming.py stark_s baseline "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py stark_s baseline "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

echo "All tracking tests completed for Stark."

# Step 3.3: OSTrack

cd ../OSTrack

# Create the default local file for OSTrack
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Copy the checkpoints to the sotas networks directory
cp -r $STARE_CKPTS_DIR/sotas/ostrack lib/test/networks

# Run frame-based tracking
echo "Starting frame-based tracking tests with OSTrack ..."
echo "------------------------------------"

# Loop through each dataset name
for dataset in "${dataset_names[@]}"; do
  echo "Running test for dataset: ${dataset}"
  python tracking/test.py ostrack esot500mix --dataset_name "${dataset}"
  echo "------------------------------------"
done

echo "All tracking tests completed."

# Run stare tracking
echo "Starting stare tracking tests with OSTrack ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options[@]}"; do
  echo "Running test for setting: ${setting}"
  python tracking/test_streaming.py ostrack esot500_baseline "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py ostrack esot500_baseline "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

echo "Starting stare tracking tests with OSTrack + C ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options[@]}"; do
  echo "Running test for setting: ${setting}"
  python tracking/test_streaming.py ostrack esot500_baseline "${setting}" --dataset_name esot500s --runid 66 --use_aas
  python tracking/streaming_eval_v4.py ostrack esot500_baseline "${setting}" --dataset_name esot500s --runid 66
  echo "------------------------------------"
done

echo "All stare tracking tests completed with OSTrack."

# Step 3.4: OSTrack + P

cd ../pred_OSTrack

# Create the default local file for OSTrack + P
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Copy the checkpoints to the sotas networks directory
cp -r $STARE_CKPTS_DIR/sotas/ostrack lib/test/networks

# Run stare tracking
echo "Starting stare tracking tests with OSTrack + P ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options[@]}"; do
  echo "Running test for setting: ${setting}"
  python tracking/test_streaming.py ostrack pred_esot500_4step "${setting}" --dataset_name esot500s --pred_next 1
  python tracking/streaming_eval_v4.py ostrack pred_esot500_4step "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

echo "Starting stare tracking tests with OSTrack + P + C ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options[@]}"; do
  echo "Running test for setting: ${setting}"
  python tracking/test_streaming.py ostrack pred_esot500_4step "${setting}" --dataset_name esot500s --pred_next 1 --runid 66 --use_aas
  python tracking/streaming_predspeed.py ostrack pred_esot500_4step "${setting}" --dataset_name esot500s --runid 66
  echo "------------------------------------"
done

echo "All stare tracking tests completed with OSTrack + P."

echo "All tasks completed successfully."


