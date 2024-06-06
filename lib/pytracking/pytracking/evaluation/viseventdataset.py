import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os
import json


class VisEventDataset(BaseDataset):
    """ VisEventDataset dataset. modified from GOT-10k DATASETS_RATIO

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split, version=None):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(self.env_settings.visEvent_dir, 'test')
        else:
            self.base_path = os.path.join(self.env_settings.visEvent_dir, 'train')

        self.sequence_list,_ = self._get_sequence_list_start(split,version)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, sequence_name,'VoxelGridComplex')
        frame_list = sorted(os.listdir(frames_path))
        # frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'visevent', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        sequence_list = []
        list_file = '{}/{}.txt'.format(self.base_path, split)
        with open (list_file,'r') as f:
            for line in f:
                sequence_list.append(split+'/'+line.strip()) 

        return sequence_list

    def _get_sequence_list_start(self, split, version=None):
        sequence_list = []
        seq2start = []
        version_list = []
        list_pair_file = '{}/{}_pair.json'.format(self.env_settings.visEvent_dir, split)
        with open(list_pair_file,'r') as j:
            seq_start_dict = json.loads(j.read())
        # build version list
        if version in ['stnet','presence','172']:
            version_list_file = '{}/{}_{}.txt'.format(self.env_settings.visEvent_dir, split, version)
            with open(version_list_file,'r') as v:
                for line in v.readlines():
                    version_list.append(line.strip())
        version_list_cp = version_list.copy()
        for seq,start_idx in seq_start_dict.items():
            if len(version_list)==0: # no version
                sequence_list.append(seq)
                seq2start.append(start_idx)
            else: # version selected
                if seq in version_list:
                    sequence_list.append(seq)
                    seq2start.append(start_idx)
                    version_list_cp.remove(seq)
        if len(version_list_cp)>0:
            print('Missing sequence')
            print(version_list_cp)
        return sequence_list, seq2start
if __name__ == '__main__':
    train_dataset = VisEventDataset('test','stnet')
    seqlist = train_dataset.get_sequence_list()
    print(seqlist)