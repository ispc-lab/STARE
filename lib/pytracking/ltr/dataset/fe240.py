import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings


class FE240(BaseVideoDataset):
    """ FE240 dataset.

    Publication:
        Object Tracking by Jointly Exploiting Frame and Event Domain
        Jiqing Zhang, Xin Yang, Yingkai Fu, Xiaopeng Wei, Baocai Yin, Bo Dong
        IEEE International Conference on Computer Vision (ICCV), July, 2021
        https://arxiv.org/pdf/2109.09052.pdf

    Download dataset from https://zhangjiqing.com/dataset/
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
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
        root = env_settings().fe240_dir if root is None else root
        super().__init__('FE240', root, image_loader)
        self.ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
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
        
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        # for seq in seq_ids:
        #     if seq not in self.sequence_list:
        #         raise(seq + ' is not in sequence list')


        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'fe240'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_sequences_in_class(self, class_name):
        raise('FE108 does not support get sequences in class')

    def _get_sequence_list(self, split):
        sequence_train = []
        sequence_test = []
        with open ('/media/group2/data/zhangzikai/FE108/fe240-list.txt','r') as f:
            f.readline()
            for line in f:
                if not 'testing set' in line:
                    sequence_train.append('train/'+line.strip()) # Different from test dataset!!
                else:
                    # f.readline()
                    break
            for line in f:
                sequence_test.append('test/'+line.strip())
        sequence_train.pop()
        if split == 'test' or split == 'val':
            return sequence_test
        elif split == 'train':
            return sequence_train
        elif split == 'total':
            return sequence_train.extend(sequence_test)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
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
            return os.path.join(seq_path, style, '{:05}.jpg'.format(frame_id+1))    # frames start from 1

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