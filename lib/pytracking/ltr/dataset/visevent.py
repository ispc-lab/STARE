import os
import os.path
from tracemalloc import start
import numpy as np
import torch
import csv
import pandas
import random
import json
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings


class VisEvent(BaseVideoDataset):
    """ VisEvent dataset.

    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, version=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the VisEvent data.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'test'. Note: The official visevent train split,
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().visEvent_dir if root is None else root
        super().__init__('VisEvent', root, image_loader)
        self.ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        # all folders inside the root
        self.seq2start=[]
        self.split = split

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            elif split in ['train', 'test']:
                # self.sequence_list = self._get_sequence_list(split)
                self.sequence_list, self.seq2start = self._get_sequence_list_start(split, version)
            else:
                raise ValueError('Unknown split name.')
            # seq_ids = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
            seq_ids = list(range(0, len(self.sequence_list)))
        elif seq_ids is None:
            raise ValueError('split and seq_ids are both None')
            self.sequence_list = self._get_sequence_list(split = 'total')
            seq_ids = list(range(0, len(self.sequence_list)))
            # seq_ids = list(range(0, len(self.sequence_list)))

        # for seq in seq_ids:
        #     if seq not in self.sequence_list:
        #         raise(seq + ' is not in sequence list')

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'visEvent'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_sequences_in_class(self, class_name):
        raise('VisEvent does not support get sequences in class')

    def _get_sequence_list_start(self, split, version=None):
        sequence_list = []
        seq2start = []
        version_list = []
        list_pair_file = '{}/{}_pair.json'.format(self.root, split)
        with open(list_pair_file,'r') as j:
            seq_start_dict = json.loads(j.read())
        if version in ['stnet']:
            version_list_file = '{}/{}_{}.txt'.format(self.root, split, version)
            with open(version_list_file,'r') as v:
                for line in v.readlines():
                    version_list.append(line.strip())
        for seq,start_idx in seq_start_dict.items():
            if len(version_list)==0: # no version
                sequence_list.append(seq)
                seq2start.append(start_idx)
            else: # version selected
                if seq in version_list:
                    sequence_list.append(seq)
                    seq2start.append(start_idx)
        return sequence_list, seq2start
        
    
    def _get_sequence_list(self, split):
        sequence_list = []
        list_file = '{}/{}.txt'.format(self.root, split)
        with open (list_file,'r') as f:
            for line in f:
                sequence_list.append(split+'/'+line.strip()) 

        return sequence_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root,self.split, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id , style='VoxelGridComplex'):
        if style == 'VoxelGridComplex':
            return os.path.join(seq_path, style, '{:05}.jpg'.format(frame_id))    

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    # def _get_vis(self, seq_path, frame_id):
    #     vis_img_list = self._get_frame_path(seq_path, frame_id, 'vis')
    #     vis_imgs = []
    #     for vii in vis_img_list:
    #         vis_imgs.append(self.image_loader(vii))
    #     return vis_imgs

    def get_frames(self, seq_id, frame_ids, anno=None):
        # offset start frame for VisEvent dataset
        start_id = self.seq2start[seq_id]
        frame_ids_fetch = [f_id + start_id for f_id in frame_ids]
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids_fetch]
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
    dataset = VisEvent(root='data/VisEvent',split='test',version='stnet')
    print(dataset)