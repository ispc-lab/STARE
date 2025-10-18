#!/bin/bash

python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_20_2
python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_20_8
python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_20_20
python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_20_50

python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_250_2
python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_250_8
python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_250_20
python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_250_50

python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_500_2
python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_500_8
python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_500_20
python tracking/test.py mixformer_convmae_online baseline --dataset_name esoth_500_50
