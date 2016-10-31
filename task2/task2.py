"""
Assignment of group Sunda slow loridae: Alexander Höreth, Sebastian Höffner, Andrea Suckro

Assignment 2 - Cats and Dogs
"""

# Task 2 - Setup
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)

# Task 3 - Implementing the network
actFun = lambda x : 1.7159 * np.tanh(2/3 * x)
actFunDeriv = lambda x : 2/3 * 1.7159 * (1 - np.tanh(2/3 * x)**2)
net = lambda x,w : actFun(actFun(x * w[0]) * w[1])
net_error = lambda y,t : 0.5 * (y - t)**2
eta = lambda c,t: c/t


# apply network
x = 2
w = np.array([1., 1.])
c = 0.3

# Task 4 - Training Data
sampleSize = 30

cats = np.random.normal(25, 5, sampleSize)
cats = np.vstack([cats, np.ones(cats.shape)])

dogs = np.random.normal(45, 15, sampleSize)
dogs = np.vstack([dogs, -1 * np.ones(dogs.shape)])

catdog = np.hstack([cats, dogs])

for t, (sample, target) in enumerate(catdog.T, 1):
    y = net(sample, w)
    w[1] = w[1] - eta(c,t) * (y - target) * actFunDeriv(actFun(w[1] * actFun(w[0] * sample)))
    w[0] = w[0] - eta(c,t) * actFunDeriv(actFun(w[0] * sample))


# Task 5 - Error function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cp = ax.plot_surface(x, y, z, cmap= plt.cm.coolwarm)
plt.colorbar(cp)
plt.show()