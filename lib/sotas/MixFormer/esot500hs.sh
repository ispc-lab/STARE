#!/bin/bash


python tracking/test_streaming.py mixformer_convmae_online baseline s101 --dataset_name esot500hs
python tracking/test_streaming.py mixformer_convmae_online baseline s102 --dataset_name esot500hs
python tracking/test_streaming.py mixformer_convmae_online baseline s107 --dataset_name esot500hs
python tracking/test_streaming.py mixformer_convmae_online baseline s103 --dataset_name esot500hs

python tracking/streaming_eval_v4.py mixformer_convmae_online baseline s101 --dataset_name esot500hs
python tracking/streaming_eval_v4.py mixformer_convmae_online baseline s102 --dataset_name esot500hs
python tracking/streaming_eval_v4.py mixformer_convmae_online baseline s107 --dataset_name esot500hs
python tracking/streaming_eval_v4.py mixformer_convmae_online baseline s103 --dataset_name esot500hs