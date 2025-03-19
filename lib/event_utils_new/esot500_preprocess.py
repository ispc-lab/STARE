import numpy as np
from dv import AedatFile
import os
from convert_event_img import *
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import argparse
from search import interpolation_search


def preprocess_esot500(path_to_esot500, fps, window, style='VoxelGridComplex'):
    transform = ToPILImage()
    anno_dir = path_to_esot500 + '/anno_t'
    anno_list = os.listdir(anno_dir)
    sequence_todo = [x.split('.')[0] for x in anno_list]
    assert 1 < window < 1e3 # ms
    assert 500 % fps == 0
    multiple = 500 // fps

    for sequence in sequence_todo:
        ae_file = path_to_esot500 + '/aedat4/{}.aedat4'.format(sequence)
        # Read events
        if os.path.exists(ae_file):
            with AedatFile(ae_file) as f:
                names = f.names
                print('Processing:',ae_file)
                events = np.hstack([packet for packet in f['events'].numpy()])
                events['timestamp'] = events['timestamp'] -events['timestamp'][0]

            save_dir = path_to_esot500 + '/{}_w{}ms/{}/VoxelGridComplex'.format(fps,window,sequence)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            label_file = path_to_esot500 + '/{}_w{}ms/{}/groundtruth.txt'.format(fps,window,sequence)
            if os.path.exists(label_file):
                print('Skipped:',sequence)
                continue
            anno_file = path_to_esot500 + '/anno_t/{}.txt'.format(sequence)
            t_anno = np.loadtxt(anno_file, delimiter=' ')
            # Downsample data
            t_anno = t_anno[::multiple] 
            label = t_anno[:,1:]
            label_time_start = t_anno[0][0]*1e6
            count = 0
            idx_label_start = 0
            idx_lable_end = len(t_anno)
            for idx in tqdm(range(len(t_anno))):
                label_time = t_anno[idx][0]*1e6
                # label = t_anno[idx][1:]
                time_left = label_time-window*1e3 # ms to us
                # Check range
                if time_left < events['timestamp'][0]:
                    idx_label_start = idx + 1
                    continue
                time_right = label_time
                if time_right > events['timestamp'][-1]:
                    idx_lable_end = idx
                    break
                idx_start = interpolation_search(events['timestamp'],time_left)
                idx_end = interpolation_search(events['timestamp'],time_right)
                event_img = convert_event_img_aedat(events[idx_start:idx_end],style)
                img = transform(event_img)
                file_name = str(count).zfill(5)+'.jpg'
                img.save(os.path.join(save_dir,file_name))
                count += 1
            label_file = path_to_esot500 + '/{}_w{}ms/{}/groundtruth.txt'.format(fps,window,sequence)
            label = label[idx_label_start:idx_lable_end]
            np.savetxt(label_file,label,fmt='%d',delimiter=',')
        else:
            print('!!! Aedat4 File Not Found:',sequence)

def main():
    parser = argparse.ArgumentParser(description='Preprocess the raw events into event frames')
    parser.add_argument('--path_to_data', type=str, default="/path/to/esot500", help="Path to ESOT500 dataset")
    parser.add_argument('--fps', type=int, default=500, help='Output frame rate.')
    parser.add_argument('--window', type=int, default=2, help='window size of each frame (ms).')
    parser.add_argument('--style', type=str, default='VoxelGridComplex', help='Event frame style.')
    parser.add_argument('--dataset_name', type=str, default='esot500', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    
    # parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    # parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    # parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    # parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
    # parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    # parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')

    args = parser.parse_args()

    if args.dataset_name == 'esot500':
        preprocess_esot500(args.path_to_data, args.fps, args.window, args.style)


if __name__ == '__main__':
    main()
