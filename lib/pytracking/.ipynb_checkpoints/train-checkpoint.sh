
# CUDA_VISIBLE_DEVICES=1 python ltr/run_training.py kys kys_fe240 > log/kys_fe240.log &
# CUDA_VISIBLE_DEVICES=2 python ltr/run_training.py tomp tomp50_fe240 > log/tomp50_fe240.log &
# CUDA_VISIBLE_DEVICES=3 python ltr/run_training.py bbreg atom_fe240 > log/atom_fe240.log

# echo "Done first 3 trackers training"

# CUDA_VISIBLE_DEVICES=1 python ltr/run_training.py dimp dimp18_fe240 > log/dimp18_fe240.log &
CUDA_VISIBLE_DEVICES=3 python ltr/run_training.py dimp prdimp18_fe240 > log/prdimp18_fe240.log

echo "Done training"
# CUDA_VISIBLE_DEVICES=3 python pytracking/run_experiment.py myexperiments esot500_offline

# free_mem = $(nvidia-smi --query-gpu=memory.free --format=csv -i 2 | grep -Eo [0-9]+)
# while [$free_mem -lt 10000 ]; do
#     free_mem = $(nvidia-smi --query-gpu=memory.free --format=csv -i 2 | grep -Eo [0-9]+)
#     sleep 10
# done
