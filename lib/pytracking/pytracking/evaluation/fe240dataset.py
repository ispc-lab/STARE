import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os


class FE240Dataset(BaseDataset):
    """ FE240 dataset. modified from GOT-10k DATASETS_RATIO

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(self.env_settings.fe240_dir, 'test')
        else:
            self.base_path = os.path.join(self.env_settings.fe240_dir, 'train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, sequence_name,'VoxelGridComplex')
        frame_list = sorted(os.listdir(frames_path))
        # frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'fe240', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        sequence_train = []
        sequence_test = []
        with open ('/media/group2/data/zhangzikai/FE108/fe240-list.txt','r') as f:
            f.readline()
            for line in f:
                if not 'testing set' in line:
                    sequence_train.append(line.strip())
                else:
                    # f.readline()
                    break
            for line in f:
                sequence_test.append(line.strip())
        sequence_train.pop()
        if split == 'test' or split == 'val':
            return sequence_test
        elif split == 'train':
            return sequence_train
if __name__ == '__main__':
    train_dataset = FE240Dataset('test')
    seqlist = train_dataset.get_sequence_list()
    print(seqlist)