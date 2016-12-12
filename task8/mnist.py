import struct
import gzip
import numpy as np
from urllib import request
from os import path, mkdir


URLS = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]


class MNIST():
    def __init__(self, dir='./data/'):
        if not path.exists('data'):
            self._fetch()
        self.testData = self._load(dir + 't10k-images-idx3-ubyte')
        self.testLabels = self._load(dir + 't10k-labels-idx1-ubyte', True)
        self.trainingData = self._load(dir + 'train-images-idx3-ubyte')
        self.trainingLabels = self._load(dir + 'train-labels-idx1-ubyte', True)
        self.allData = np.concatenate((self.testData, self.trainingData))
        
    def _load(self, path, labels=False):
        with open(path, 'rb') as fd:
            magic, numberOfItems = struct.unpack('>ii', fd.read(8))
            if (not labels and magic != 2051) or (labels and magic != 2049):
                raise LookupError('Not a MNIST file')
            if not labels:
                rows, cols = struct.unpack('>II', fd.read(8))
                images = np.fromfile(fd, dtype='uint8')
                images = images.reshape((numberOfItems, rows, cols))
                return images
            else:
                labels = np.fromfile(fd, dtype='uint8')
                return labels

    def _extract(self, src, dst):
        with gzip.open(src, 'rb') as infile:
            with open(dst, 'wb') as outfile:
                for line in infile:
                    outfile.write(line)

    def _fetch(self):
        print('Fetching data...')
        mkdir('data')
        for url in URLS:
            target = path.join('data', path.basename(url))
            print(target)
            request.urlretrieve(url, target)
            self._extract(target, path.splitext(target)[0])
        print('Done fetching data.')

    def getTrainingBatch(self, n=100):
        sample = np.random.random_integers(0, len(self.trainingData)-1, n)
        return self.trainingData[sample], self.trainingLabels[sample]

    def getAllDataBatch(self, n=100):
        sample = np.random.random_integers(0, len(self.allData)-1, n)
        return self.allData[sample]