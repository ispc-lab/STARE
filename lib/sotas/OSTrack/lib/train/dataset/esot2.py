import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings



class ESOT2(BaseVideoDataset):
    """ EventSOT2 dataset.

    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None, inter=None, window=None):
        """
        args:
            root - path to the fe108 data.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'test'. Note: The official fe108 train split,
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().esot2_dir if root is None else root
        super().__init__('ESOT2', root, image_loader)
        self.ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        self.inter = inter
        self.window = window
        if inter or window:
            self.root = os.path.join(root, f'inter{inter}_w{window}ms')
        else:
            self.root = os.path.join(root, 'default')
        # all folders inside the root

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            elif split in ['train', 'test']:
                self.sequence_list = self._get_sequence_list(split)
            else:
                raise ValueError('Unknown split name.')
            # seq_ids = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
            seq_ids = list(range(0, len(self.sequence_list)))
        elif seq_ids is None:
            self.sequence_list = self._get_sequence_list(split = 'total')
            seq_ids = list(range(0, len(self.sequence_list)))
            # seq_ids = list(range(0, len(self.sequence_list)))

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        if self.inter or self.window:
            return f'EventSOT2_inter{self.inter}_w{self.window}ms'
        return 'EventSOT2'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_sequences_in_class(self, class_name):
        raise('EventSOT2 does not support get sequences in class')

    def _get_sequence_list(self, split):
        seq_list = []
        with open ('data/EventSOT/EventSOT2/{}.txt'.format(split),'r') as f:
            for line in f:
                    seq_list.append(line.strip())
        return seq_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        if gt.shape[1]==5:
            gt = gt[:,1:]
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id , style='VoxelGridComplex'):
        if style == 'VoxelGridComplex':
            return os.path.join(seq_path, style, '{:05}.jpg'.format(frame_id))    # frames start from 0

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    # def _get_vis(self, seq_path, frame_id):
    #     vis_img_list = self._get_frame_path(seq_path, frame_id, 'vis')
    #     vis_imgs = []
    #     for vii in vis_img_list:
    #         vis_imgs.append(self.image_loader(vii))
    #     return vis_imgs

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        # vis_list = [self._get_vis(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = None

        return frame_list, anno_frames, object_meta #, vis_list
    
    def get_sequence_name(self, seq_id):
        return self.sequence_list[seq_id]

if __name__ == '__main__':
    # dataset = ESOT500(root='data/EventSOT/EventSOT500/EventSOT500/pre500',split='test')

    print(dataset)