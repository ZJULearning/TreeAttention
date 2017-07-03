import torch as th
import six.moves.cPickle as Pickle
from torch.utils.data import Dataset
import numpy as np


class QADataset(Dataset):
    def __init__(self, filepath):
        self.data = None
        with open(filepath, 'rb') as datafile:
            self.data = Pickle.load(datafile)

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = self.data[index]
        vid = sample[0]
        vf = Pickle.load(open(
            '/home/xuehongyang/youtubeclips-dataset/video_features/fea%s.pkl' %
            vid[3:],
            'rb'),
                         encoding='latin1')

        return th.from_numpy(np.asarray(vf, dtype=np.float32)),\
            sample[1],\
            th.from_numpy(np.asarray([sample[2]], dtype=np.int64))
