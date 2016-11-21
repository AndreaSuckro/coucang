from os import path
import pickle
import numpy as np
from numpy.random import shuffle, random_integers


class Data():
    def __init__(self, fdir='./data/', split=(70, 20, 10)):
        self.data = self._load(path.join(fdir, 'trainData.pickle'))
        self.labels = self._load(path.join(fdir, 'trainLabels.pickle'))
        self.train, self.validation, self.test = \
            self._split(split, self.data, self.labels)

    def _load(self, fpath):
        with open(fpath, 'rb') as f:
            return pickle.load(f)

    def _split(self, split, data, labels):
        idx = np.arange(len(labels))
        shuffle(idx)
        split = (len(idx) // split[0], len(idx) // split[1])
        idx = np.split(idx, split)
        train = (data[idx[0]], labels[idx[0]])
        validation = (data[idx[1]], labels[idx[1]])
        test = (data[idx[2]], labels[idx[2]])
        return train, validation, test

    def getBatch(self, n=100, src=False):
        src = src or self.train
        data, labels = src
        sample = random_integers(0, len(labels)-1, n)
        return data[sample], labels[sample]
