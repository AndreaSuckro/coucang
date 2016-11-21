from matplotlib import pyplot as plt
from data import Data
from numpy import squeeze

data = Data()

plt.suptitle('Data Samples')
for i, (img, label) in enumerate(zip(*data.get_batch(5*12, data.test))):
    ax = plt.subplot(5, 12, 1+i)
    ax.set_title(label)
    ax.axis('off')
    ax.imshow(squeeze(img), cmap='gray')
plt.show()
