# CUDA_VISIBLE_DEVICES=1 python pytracking/run_experiment.py myexperiments esotVH_offline

# CUDA_VISIBLE_DEVICES=1 python pytracking/run_experiment.py myexperiments esotVM_offline

# CUDA_VISIBLE_DEVICES=1 python pytracking/run_experiment.py myexperiments esotVL_offline

# CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments esot500_fps_window_fe > log/esot500_fps_window_fe.log &
# CUDA_VISIBLE_DEVICES=1 python pytracking/run_experiment.py myexperiments esot2_interp_fe

# CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py offline_evaluation offline_ultimate

# CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment_streaming.py exp_streaming streaming_18
CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment_streaming.py exp_streaming streaming_34
CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment_streaming.py exp_streaming streaming_35

# CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment_streaming.py exp_streaming streaming_egt
# CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment_streaming.py exp_streaming streaming_egt_3

# CUDA_VISIBLE_DEVICES=2 python eval/streaming_eval_v3.py exp_streaming streaming_3

# CUDA_VISIBLE_DEVICES=2 python eval/streaming_eval_v3.py exp_streaming streaming_18
CUDA_VISIBLE_DEVICES=0 python eval/streaming_eval_v3.py exp_streaming streaming_34
CUDA_VISIBLE_DEVICES=0 python eval/streaming_eval_v3.py exp_streaming streaming_35


# CUDA_VISIBLE_DEVICES=2 python eval/streaming_eval_v3.py exp_streaming streaming_egt
# CUDA_VISIBLE_DEVICES=2 python eval/streaming_eval_v3.py exp_streaming streaming_egt_3


# free_mem = $(nvidia-smi --query-gpu=memory.free --format=csv -i 2 | grep -Eo [0-9]+)
# while [$free_mem -lt 10000 ]; do
#     free_mem = $(nvidia-smi --query-gpu=memory.free --format=csv -i 2 | grep -Eo [0-9]+)
#     sleep 10
# done
