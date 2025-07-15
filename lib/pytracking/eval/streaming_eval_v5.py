'''
Streaming evaluation
Given real-time tracking outputs,
it pairs them with the ground truth.

Note that this script does not need to run in real-time
'''

import sys
import os
import importlib
import argparse
import pickle

import numpy as np

from os.path import join, isfile
from tqdm import tqdm


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)


def find_last_pred(gt_t, pred_raw):
    pred_timestamps = pred_raw['out_timestamps']
    pred_timestamps[0] = 0
    gt_t = gt_t * 1e6
    # print(gt_t, pred_timestamps[-1])
    # assert abs(gt_t - pred_timestamps[-1]) < 100  # time unit:s

    last_pred_idx = np.searchsorted(pred_timestamps, gt_t) - 1
    pred_results = pred_raw['results_raw']
    pred_last_result = pred_results[last_pred_idx]
    pred_last_time = pred_timestamps[last_pred_idx]

    assert pred_last_time <= gt_t
    # print(gt_t, pred_last_time)

    return pred_last_result

def stream_eval(gt_anno_t: list, raw_result: dict):
    pred_final = []
    for line in gt_anno_t:
        gt_t = line[0]
        pred_label = find_last_pred(gt_t, raw_result)
        pred_bbox = pred_label
        pred_final.append(pred_bbox)

    return pred_final

def eval_sequence_stream(sequence, tracker, stream_setting, sim_frame=False, fps=500, window=2):
    print("Stream Evaluation: {} {} {} {}".format(sequence.name, tracker.name, tracker.run_id, stream_setting.id))

    tracker_name = tracker.name
    param = tracker.parameter_name
    gt_anno_t = sequence.ground_truth_t

    if sim_frame:
        assert 500 % fps == 0, "fps should be a divisor of 500"
        multiple = 500 // fps
        gt_anno_t = gt_anno_t[::multiple]

    save_dir = os.path.join(tracker.results_dir_rt_final, str(stream_setting.id))
    if sim_frame:
        save_dir = tracker.results_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # if os.path.exists(os.path.join(save_dir, sequence.name+'.txt')):
    #     print('Already exists. Skipped. ')
    #     return

    raw_result = pickle.load(open(os.path.join(tracker.results_dir_rt, str(stream_setting.id), sequence.name + '.pkl'), 'rb'))
    assert raw_result['stream_setting'] == stream_setting.id

    pred_final = stream_eval(gt_anno_t, raw_result)
    if sim_frame:
        np.savetxt('{}/{}_{}_w{}ms.txt'.format(save_dir, sequence.name[:-2], fps, window), pred_final, fmt='%d', delimiter='\t')
    else:
        np.savetxt('{}/{}.txt'.format(save_dir, sequence.name), pred_final, fmt='%d', delimiter='\t')

def run_streaming_eval(experiment_module: str, experiment_name: str, sim_frame=False, fps=500, window=2, debug=0, threads=0):
    print('Running:  {}  {}'.format(experiment_module, experiment_name))
    if sim_frame:
        print(f'Simulation frame evaluation mode enabled, FPS: {fps}, Window: {window} ms')

    expr_module = importlib.import_module('pytracking.experiments.{}'.format(experiment_module))
    expr_func = getattr(expr_module, experiment_name)

    trackers, dataset, stream_setting = expr_func()
    for seq in dataset:
        for tracker_info in trackers:
            eval_sequence_stream(seq, tracker_info, stream_setting, sim_frame, fps, window)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')

    parser.add_argument('experiment_module', type=str, help='Name of experiment module in the experiments/ folder.')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment function.')
    parser.add_argument('--sim_frame', action='store_true', help="For simulated frame-based evaluation, not real-time tracking.")
    parser.add_argument('--fps', type=int, default=500, help='Frame rate of simulated frame-based evaluation.')
    parser.add_argument('--window', type=int, default=2, help='Window size of each frame (ms) for simulated frame-based evaluation.')

    args = parser.parse_args()

    run_streaming_eval(args.experiment_module, args.experiment_name, args.sim_frame, args.fps, args.window)


if __name__ == '__main__':
    main()
