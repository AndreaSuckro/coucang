from urllib import request
from os import path, mkdir
from shutil import unpack_archive
import gzip
import glob

URLS = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]


def extract(src, dst):
    with gzip.open(src, 'rb') as infile:
      with open(dst, 'wb') as outfile:
          for line in infile:
              outfile.write(line)


if not path.exists('data'):
    mkdir('data')

for url in URLS:
    target = path.join('data', path.basename(url))
    print(target)
    request.urlretrieve(url, target)
    extract(target, path.splitext(target)[0])

print('done')
