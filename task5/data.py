from os import path
import pickle
import numpy as np
from numpy.random import permutation, random_integers


class Data():
    def __init__(self, fdir='./data/', split=(70, 20, 10)):
        self.data = self._load(path.join(fdir, 'trainData.pickle'))
        self.labels = self._load(path.join(fdir, 'trainLabels.pickle')) % 10
        self.train, self.validation, self.test = \
            self._split(split, self.data, self.labels)

    def _load(self, fpath):
        with open(fpath, 'rb') as f:
            return pickle.load(f)

    def _split(self, split, data, labels):
        idx = permutation(np.arange(len(labels)))
        parts = ()
        for s in split:
            left = len(parts[-1][0]) if len(parts) else 0
            right = left + len(idx) // 100 * s
            parts += ((data[left:right], labels[left:right]),)
        return parts

    def get_batch(self, n=100, src=False):
        src = src or self.train
        data, labels = src
        sample = random_integers(0, len(labels)-1, n)
        return data[sample], labels[sample]
