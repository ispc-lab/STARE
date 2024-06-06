import numpy as np
from pytracking.evaluation.data import Sequence,SequenceEvent, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os
from dv import AedatFile


class ESOT500DatasetStream(BaseDataset):
    """ ESOT500 Streaming dataset. modified from GOT-10k DATASETS_RATIO

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    '''JieChu:
    The param annot_fps is added by JieChu.
    And this param is added for the annotation alignment and evaluation 
    with lower frequence annotation than 500fps.
    '''
    def __init__(self, split,annot_fps=500):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = self.env_settings.esot500_dir
        # FPS only for gt evaluation

        self.sequence_list = self._get_sequence_list(split)
        self.split = split
        self.annot_fps = annot_fps

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path,500, sequence_name)

        '''JieChu:
        This ground_truth_rect is also infuenced by str(self.annot_fps).
        the annotTrans_500toFPS function will implement the influential.
        '''
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        ground_truth_rect = self.annotTrans_500toFPS(ground_truth_rect,self.annot_fps)
        # ground_truth_rect = ground_truth_rect[::25]

        '''JieChu:
        This frames_list is temporarily not infuenced by str(self.annot_fps).
        So there's no annotTrans_500toFPS called.
        '''
        frames_path = '{}/{}/{}'.format(self.base_path,500, sequence_name,'VoxelGridComplex')
        frame_list = sorted(os.listdir(frames_path))
        # frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        
        aeFile = os.path.join(self.base_path, 'aedat4',sequence_name+'.aedat4')
        
        '''JieChu:
        This t_anno_path is also infuenced by str(self.annot_fps).
        the annotTrans_500toFPS function will implement the influential.
        '''
        t_anno_path = os.path.join(self.base_path, 'anno_t',sequence_name+'.txt')
        t_anno = np.loadtxt(t_anno_path, delimiter=' ')
        t_anno=self.annotTrans_500toFPS(t_anno,self.annot_fps)

        sequence_name = sequence_name+'_s'
        dataset_name = 'esot500s'

        return SequenceEvent(sequence_name, frames_list, aeFile, 
                             dataset_name, ground_truth_rect.reshape(-1, 4), ground_truth_t=t_anno)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        # training / testing split
        seq_list = []
        # with open ('/home/test4/code/OSTrack/data/EventSOT/EventSOT500/EventSOT500/{}.txt'.format(split),'r') as f:
        with open ('{}/{}.txt'.format(self.base_path,split),'r') as f:
            for line in f:
                    seq_list.append(line.strip())
        return seq_list

    def annotTrans_500toFPS(self,gt_anno_t,FPS:int):
        if FPS == 500:
            return gt_anno_t

        newAnnot=[]
        index=0.0 #float
        stride=500/FPS #float
        while int(index) < len(gt_anno_t):
            newAnnot.append(gt_anno_t[int(index)])
            index+=stride
        return np.array(newAnnot, dtype=gt_anno_t.dtype)

if __name__ == '__main__':
    train_dataset = ESOT500Dataset('test','500')
    seqlist = train_dataset.get_sequence_list()
    print(seqlist)