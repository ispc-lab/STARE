import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class ESOT500Dataset(BaseDataset):
    """ EventCarla dataset. modified from GOT-10k DATASETS_RATIO

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split, fps=500, window=2):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = self.env_settings.esot500_dir
        self.base_path = os.path.join(self.base_path,'{}_w{}ms'.format(fps,window))
        print('Using subset:{} fps , {} ms window for testing'.format(fps,window))
        self.sequence_list = self._get_sequence_list(split)
        self.split = split
        self.fps = fps
        self.window = window

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, sequence_name,'VoxelGridComplex')
        frame_list = sorted(os.listdir(frames_path))
        # frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name+'_{}_w{}ms'.format(self.fps, self.window), frames_list, 'esot_{}_w{}ms'.format(self.fps,self.window), ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        # training / testing split
        seq_list = []
        with open (self.env_settings.esot500_dir + '/{}.txt'.format(split),'r') as f:
            for line in f:
                    seq_list.append(line.strip())
        return seq_list

if __name__ == '__main__':
    train_dataset = ESOT500Dataset('test',500,4)
    seqlist = train_dataset.get_sequence_list()
    print(seqlist)
