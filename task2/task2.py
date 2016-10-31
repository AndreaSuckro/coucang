"""
Assignment of group Sunda slow loridae: Alexander Höreth, Sebastian Höffner, Andrea Suckro

Assignment 2 - Cats and Dogs
"""

# Task 2 - Setup
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Task 3 - Implementing the network


# Task 4 - Training Data
sampleSize = 30
np.random.seed(1)
cats = np.random.normal(25, 5, sampleSize)
dogs = np.random.normal(45, 15, sampleSize)

# Task 5 - Error function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cp = ax.plot_surface(X, Y, Z, cmap= plt.cm.coolwarm)
plt.colorbar(cp)
plt.show()