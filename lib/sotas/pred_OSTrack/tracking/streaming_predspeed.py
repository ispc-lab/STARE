'''
Streaming predict speed
Given real-time tracking outputs, 
and pairs them with the ground truth.

Note that this script does not need to run in real-time
'''

import argparse, pickle
from os.path import join, isfile
import numpy as np
from tqdm import tqdm
import sys
import os

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.environment import env_settings

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('stream_setting', type=str, help='Name of stream_setting file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='esot500s', help='Name of dataset.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--dynamic_order', type=int, default=1, choices=[1, 2, 3], help='dynamic order of speed compensation')

    args = parser.parse_args()

    return args

def find_last_pred(gt_t, pred_raw, t0):
    pred_timestamps = pred_raw['out_timestamps']
    pred_timestamps[0] = 0
    in_timestamps = pred_raw['in_timestamps']
    gt_t = gt_t * 1e6
    # print(gt_t, pred_timestamps[-1])
    # assert abs(gt_t - pred_timestamps[-1]) < 100  # time unit:s

    last_pred_idx = np.searchsorted(pred_timestamps, gt_t) - 1
    pred_results = pred_raw['results_raw']
    pred_last_result = pred_results[last_pred_idx]
    in_time = in_timestamps[last_pred_idx]

    bbox_speeds = pred_raw['bbox_speed']
    bbox_speeds = [[0, 0, 0, 0]] + bbox_speeds
    bbox_speed = bbox_speeds[last_pred_idx]

    # speed(cx,cy,w,h)->(x,y,w,h)
    bbox_speed[0] = bbox_speed[0] - 0.5 * bbox_speed[2]
    bbox_speed[1] = bbox_speed[1] - 0.5 * bbox_speed[3]

    # print(gt_t, pred_last_time)
    return in_time,pred_last_result,bbox_speed

def main():
    args = parse_args()
    # trackers = os.listdir(args.raw_root)

    #######################################################
    trackers = [args.tracker_name]
    params = [args.tracker_param]
    s_id = args.stream_setting[1:]
    run_id = args.runid
    #######################################################

    env = env_settings()
    gt_path = env.esot500_dir + '/anno_t/'
    raw_result_path = env.results_path_rt
    save_path = env.results_path_rt_final
    
    gt_list = os.listdir(gt_path)
    gt_list = [os.path.join(gt_path, i) for i in os.listdir(gt_path) if i.endswith('.txt')]
    for tracker in trackers:
        for param in params:
            if run_id is None:
                raw_result_list_dir = '{}/{}/{}/{}'.format(raw_result_path, tracker, param, str(s_id))
            else:
                raw_result_list_dir = '{}/{}/{}_{:03d}/{}'.format(raw_result_path, tracker, param, run_id, str(s_id))

            raw_result_list = os.listdir(raw_result_list_dir)
            for sequence in tqdm(raw_result_list):
                sequence_name = sequence.split('_')[0]
                raw_result = pickle.load(open(os.path.join(raw_result_list_dir, sequence), 'rb'))
                gt_anno_t_path = os.path.join(gt_path, sequence_name+'.txt')
                gt_anno_t = np.loadtxt(gt_anno_t_path, delimiter=' ')

                pred_final = []
                last_time = gt_anno_t[0][0] # the first gt_t
                ii = 0

                v_last = None
                a_last = None
                for line in gt_anno_t:
                    gt_t = line[0]
                    in_time, pred_label, bbox_speed = find_last_pred(gt_t, raw_result, gt_anno_t[0][0])
                    pred_time = in_time

                    # raw result
                    pred_bbox = np.array(pred_label).reshape(-1, 4)
                    delta_t = (gt_t * 1e6 - in_time) / 1e3

                    # first order
                    v_curr = np.array(bbox_speed).reshape(-1, 4)
                    pred_bbox += v_curr * delta_t

                    # second order
                    if args.dynamic_order >= 2 and v_last is not None:
                        a_curr = v_curr - v_last
                        pred_bbox += 0.5 * a_curr * (delta_t) ** 2

                        # third order
                        if args.dynamic_order >= 3 and a_last is not None:
                            j_curr = a_curr - a_last
                            pred_bbox += (1 / 6) * j_curr * (delta_t) ** 3
                            
                        a_last = a_curr

                    v_last = v_curr

                    # pred_bbox = np.array(pred_label).reshape(-1, 4)
                    if pred_bbox[0, 2] <= 0:
                        pred_bbox[0, 2] = 1
                    if pred_bbox[0, 3] <= 0:
                        pred_bbox[0, 3] = 1

                    pred_final.append(pred_bbox)

                if run_id is None:
                    save_dir = '{}/{}/{}/{}'.format(save_path, tracker, param, str(s_id))
                else:
                    save_dir = '{}/{}/{}_{:03d}/{}'.format(save_path, tracker, param, run_id, str(s_id))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                pred_final = np.stack(pred_final, 0).reshape(-1, 4)
                np.savetxt('{}/{}_s.txt'.format(save_dir, sequence_name),
                           pred_final, fmt='%d', delimiter='\t')
    
        mismatch = 0
        fps_a=[]

if __name__ == '__main__':
    main()
