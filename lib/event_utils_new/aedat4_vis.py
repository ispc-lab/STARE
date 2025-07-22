import os
import argparse
import numpy as np
import itertools

from torchvision.transforms import ToPILImage
from search import interpolation_search
from dv import AedatFile
from convert_event_img import *


def preprocess_aedat4(path_to_dir, fps, window, width, height, style='VoxelGridComplex'):
    transform = ToPILImage()
    file_list = os.listdir(path_to_dir + "/aedat4")
    sequence_todo = [x.split('.')[0] for x in file_list]
    # assert 1 < window < 1e3  # ms

    for sequence in sequence_todo:
        ae_file = path_to_dir + '/aedat4/{}.aedat4'.format(sequence)

        # Read events
        if os.path.exists(ae_file):
            with AedatFile(ae_file) as f:
                names = f.names
                print('Processing:', ae_file)

                # Read events from the file
                # events = np.hstack([packet for packet in f['events'].numpy()])

                # testing for subset of large files
                events_iterator = f['events'].numpy()
                events = np.hstack([packet for packet in itertools.islice(events_iterator, 10000)])

                events['timestamp'] = events['timestamp'] - events['timestamp'][0]

            save_dir = path_to_dir + '/{}_w{}ms/{}/VoxelGridComplex'.format(fps, window, sequence)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            count = 0
            time_left = 0
            while True:
                time_left += 1000 / fps * 1e3  # ms to us
                time_right = time_left + window * 1e3  # ms to us
                if time_right > events['timestamp'][-1]:
                    break

                idx_start = interpolation_search(events['timestamp'], time_left)
                idx_end = interpolation_search(events['timestamp'], time_right)
                event_img = convert_event_img_aedat(events[idx_start:idx_end], style, height, width)

                img = transform(event_img)
                file_name = str(count).zfill(5) + '.jpg'
                img.save(os.path.join(save_dir, file_name))
                count += 1
        else:
            print('!!! Aedat4 File Not Found:', sequence)


def main():
    parser = argparse.ArgumentParser(description='Preprocess the raw events into event frames')

    parser.add_argument('--path_to_data', type=str, default="/path/to/esot500", help="Path to ESOT500 dataset")
    parser.add_argument('--fps', type=int, default=500, help='Output frame rate.')
    parser.add_argument('--window', type=int, default=2, help='window size of each frame (ms).')
    parser.add_argument('--height', type=int, default=260, help='Height of the event frame.')
    parser.add_argument('--width', type=int, default=346, help='Width of the event frame.')
    parser.add_argument('--style', type=str, default='VoxelGridComplex', help='Event frame style.')

    args = parser.parse_args()

    preprocess_aedat4(args.path_to_data, args.fps, args.window, args.width, args.height, args.style)


if __name__ == '__main__':
    main()
