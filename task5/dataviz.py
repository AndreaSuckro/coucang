from matplotlib import pyplot as plt
import numpy as np


def plot_samples(data, size=(5, 12)):
    """Plots some random samples.

    Args:
        data: a Data object
        size: the number of samples to plot (rows, columnss)
    """
    plt.figure()
    plt.suptitle('Random Data Samples')
    for i, (img, label) in enumerate(zip(*data.get_batch(np.multiply(*size), data.test))):
        ax = plt.subplot(size[0], size[1], 1+i)
        ax.set_title(label)
        ax.axis('off')
        ax.imshow(np.squeeze(img), cmap='gray')
    plt.show()


