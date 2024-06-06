import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os


class ESOT2Dataset(BaseDataset):
    """ ESOT2Dataset. modified from GOT-10k DATASETS_RATIO

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split, interpolate, window):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = self.env_settings.esot2_dir
        if interpolate!=1:
            self.base_path = os.path.join(self.base_path, 'inter{}_w{}ms'.format(interpolate,window))
        else:
            self.base_path = os.path.join(self.base_path, 'default')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split
        self.interpolate = interpolate
        self.window = window

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        if ground_truth_rect.shape[1]==5:
            ground_truth_rect = ground_truth_rect[:,1:]

        frames_path = '{}/{}/{}'.format(self.base_path, sequence_name,'VoxelGridComplex')
        frame_list = sorted(os.listdir(frames_path))
        # frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        if self.interpolate!=1:
            sequence_name = sequence_name+'_inter{}_w{}ms'.format(self.interpolate, self.window)
            dataset_name = 'esot2_inter{}_w{}ms'.format(self.interpolate, self.window)
        else:
            sequence_name = sequence_name+'_default'
            dataset_name = 'esot2_default'
        return Sequence(sequence_name, frames_list, dataset_name, ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        seq_list = []
        with open ('/home/test4/code/OSTrack/data/EventSOT/EventSOT2/{}.txt'.format(split),'r') as f:
            for line in f:
                    seq_list.append(line.strip())
        return seq_list

# if __name__ == '__main__':
    # train_dataset = ESOT500Dataset('test')
    # seqlist = train_dataset.get_sequence_list()
    # print(seqlist)