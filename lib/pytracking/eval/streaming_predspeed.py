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

# the line below is for running in both the current directory 
# and the repo's root directory

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', default='/home/test4/code/EventBenchmark/lib/pytracking/pytracking/output/tracking_results_rt_raw',type=str,
        help='raw result root')
    parser.add_argument('--target_root', default='/home/test4/code/EventBenchmark/lib/pytracking/pytracking/output/tracking_results_rt_final',type=str,
        help='target result root')
    parser.add_argument('--gt_root',default='/home/test4/code/EventBenchmark/data/EventSOT500/anno_t/', type=str)
    # parser.add_argument('--fps', type=float, default=30)
    # parser.add_argument('--eta', type=float, default=0, help='eta >= -1')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()
    return args

def find_last_pred(gt_t, pred_raw, t0):
    pred_timestamps = pred_raw['out_timestamps']
    pred_timestamps[0] = 0
    in_timestamps = pred_raw['in_timestamps']
    gt_t = gt_t*1e6
    # print(gt_t, pred_timestamps[-1])
    # assert abs(gt_t - pred_timestamps[-1]) < 100  # time unit:s
    last_pred_idx = np.searchsorted(pred_timestamps, gt_t)-1
    pred_results = pred_raw['results_raw']
    pred_last_result = pred_results[last_pred_idx]
    in_time = in_timestamps[last_pred_idx]
    bbox_speeds = pred_raw['bbox_speed']
    bbox_speeds = [[0,0,0,0]]+bbox_speeds
    bbox_speed = bbox_speeds[last_pred_idx]
    # speed(cx,cy,w,h)->(x,y,w,h)
    bbox_speed[0] = bbox_speed[0]-0.5*bbox_speed[2]
    bbox_speed[1] = bbox_speed[1]-0.5*bbox_speed[3]

    # print(gt_t, pred_last_time)
    return in_time,pred_last_result,bbox_speed

def main():
    args = parse_args()
    # trackers = os.listdir(args.raw_root)
    trackers = ['ostrack']
    params = ['pred_esot500mix_bs4','pred_esot500_bs4']
    id = 14
    gt_path = args.gt_root
    raw_result_path = args.raw_root
    save_path = args.target_root
    
    gt_list = os.listdir(gt_path)
    gt_list = [os.path.join(gt_path, i) for i in os.listdir(gt_path) if i.endswith('.txt')]
    for tracker in trackers:
        for param in params:
            raw_result_list_dir = os.path.join(raw_result_path,tracker,param,str(id))
            raw_result_list = os.listdir(raw_result_list_dir)
            for sequence in tqdm(raw_result_list):
                sequence_name = sequence.split('_')[0]
                raw_result = pickle.load(open(os.path.join(raw_result_list_dir, sequence), 'rb'))
                gt_anno_t_path = os.path.join(gt_path, sequence_name+'.txt')
                gt_anno_t = np.loadtxt(gt_anno_t_path, delimiter=' ')

                pred_final = []
                last_time = gt_anno_t[0][0] # the first gt_t
                ii = 0

                for line in gt_anno_t:
                    gt_t = line[0]
                    in_time, pred_label, bbox_speed = find_last_pred(gt_t, raw_result,gt_anno_t[0][0])
                    pred_time = in_time

                    # pred_bbox = np.array(pred_label).reshape(-1,4) + np.array(bbox_speed).reshape(-1,4)*(gt_t*1e6-in_time)/1e3
                    pred_bbox = np.array(pred_label).reshape(-1,4)
                    if pred_bbox[0,2]<=0:
                        pred_bbox[0,2]=1
                    if pred_bbox[0,3]<=0:
                        pred_bbox[0,3]=1
                    pred_final.append(pred_bbox)
                
                save_dir = '{}/{}/{}/{}'.format(save_path,tracker,param,str(id))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                pred_final = np.stack(pred_final,0).reshape(-1,4)
                np.savetxt('{}/{}/{}/{}/{}_s.txt'.format(save_path,tracker,param,str(id),sequence_name),pred_final,fmt='%d',delimiter='\t')
    
        mismatch = 0
        fps_a=[]
if __name__ == '__main__':
    main()