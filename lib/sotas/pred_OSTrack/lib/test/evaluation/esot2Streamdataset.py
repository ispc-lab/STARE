import numpy as np
from lib.test.evaluation.data import Sequence,SequenceEvent, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class ESOT2DatasetStream(BaseDataset):
    """ ESOT2 Streaming Dataset. modified from GOT-10k DATASETS_RATIO

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split, interpolate=1):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = self.env_settings.esot2_dir

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, 'default',sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        if ground_truth_rect.shape[1]==5:
            ground_truth_rect = ground_truth_rect[:,1:]

        frames_path = '{}/{}/{}/{}'.format(self.base_path,'default',sequence_name,'VoxelGridComplex')
        frame_list = sorted(os.listdir(frames_path))
        # frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        aeFile = os.path.join(self.base_path, 'aedat4',sequence_name+'.aedat4')
        t_anno_path = os.path.join(self.base_path, 'anno_t',sequence_name+'.txt')
        t_anno = np.loadtxt(t_anno_path, delimiter=' ')

        sequence_name = sequence_name+'_s'
        dataset_name = 'esot2s'

        return SequenceEvent(sequence_name, frames_list, aeFile,
                             dataset_name, ground_truth_rect.reshape(-1, 4),
                             ground_truth_t=t_anno)

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