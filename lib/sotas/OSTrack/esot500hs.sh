#!/bin/bash

python tracking/test_streaming.py ostrack esot500_baseline s101 --dataset_name esot500hs
python tracking/test_streaming.py ostrack esot500_baseline s107 --dataset_name esot500hs
python tracking/test_streaming.py ostrack esot500_baseline s102 --dataset_name esot500hs
python tracking/test_streaming.py ostrack esot500_baseline s103 --dataset_name esot500hs

python tracking/streaming_eval_v4.py ostrack esot500_baseline s101 --dataset_name esot500hs
python tracking/streaming_eval_v4.py ostrack esot500_baseline s107 --dataset_name esot500hs
python tracking/streaming_eval_v4.py ostrack esot500_baseline s102 --dataset_name esot500hs
python tracking/streaming_eval_v4.py ostrack esot500_baseline s103 --dataset_name esot500hs