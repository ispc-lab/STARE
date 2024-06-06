'''
Streaming evaluation
Given real-time tracking outputs,
it pairs them with the ground truth.

Note that this script does not need to run in real-time
'''

import argparse, pickle
from os.path import join, isfile
import numpy as np
from tqdm import tqdm
import sys
import os
import importlib

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)


def find_last_pred(gt_t, pred_raw):
    pred_timestamps = pred_raw['out_timestamps']
    pred_timestamps[0] = 0
    gt_t = gt_t*1e6
    # print(gt_t, pred_timestamps[-1])
    # assert abs(gt_t - pred_timestamps[-1]) < 100  # time unit:s
    last_pred_idx = np.searchsorted(pred_timestamps, gt_t)-1
    pred_results = pred_raw['results_raw']
    pred_last_result = pred_results[last_pred_idx]
    pred_last_time = pred_timestamps[last_pred_idx]
    assert pred_last_time <= gt_t
    # print(gt_t, pred_last_time)
    return pred_last_result

def stream_eval(gt_anno_t:list, raw_result:dict):
    pred_final = []
    for line in gt_anno_t:
        gt_t = line[0]
        pred_label = find_last_pred(gt_t, raw_result)
        pred_bbox = pred_label
        pred_final.append(pred_bbox)
    return pred_final

def eval_sequence_stream(sequence, tracker, stream_setting):

    print("Stream Evaluation: {} {} {} {}".format(sequence.name, tracker.name, tracker.run_id, stream_setting.id))
    tracker_name = tracker.name
    param = tracker.parameter_name
    gt_anno_t = sequence.ground_truth_t
    # gt_anno_t = gt_anno_t[::25] # downsample annotation frequency
    save_dir = os.path.join(tracker.results_dir_rt_final,str(stream_setting.id))
    if tracker.run_id != None:
        save_dir = os.path.join(tracker.results_dir_rt_final,str(stream_setting.id))
        # save_dir = os.path.join(tracker.results_dir_rt_final,'18d') # temporal
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if os.path.exists(os.path.join(save_dir, sequence.name+'.txt')):
    #     print('Already exists. Skipped. ')
    #     return

    raw_result = pickle.load(open(os.path.join(tracker.results_dir_rt, str(stream_setting.id), sequence.name+'.pkl'), 'rb'))
    assert raw_result['stream_setting'] == stream_setting.id
    pred_final = stream_eval(gt_anno_t, raw_result)

    np.savetxt('{}/{}.txt'.format(save_dir,sequence.name),pred_final,fmt='%d',delimiter='\t')

    
def run_streaming_eval(experiment_module: str, experiment_name: str, debug=0, threads=0):
    expr_module = importlib.import_module('pytracking.experiments.{}'.format(experiment_module))
    expr_func = getattr(expr_module, experiment_name)
    trackers, dataset, stream_setting = expr_func()
    print('Running:  {}  {}'.format(experiment_module, experiment_name))
    for seq in dataset:
        for tracker_info in trackers:
                eval_sequence_stream(seq, tracker_info, stream_setting)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('experiment_module', type=str, help='Name of experiment module in the experiments/ folder.')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment function.')

    args = parser.parse_args()

    run_streaming_eval(args.experiment_module, args.experiment_name)

if __name__ == '__main__':
    main()
