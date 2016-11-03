"""
Assignment of group Sunda slow loridae: Alexander Höreth, Sebastian Höffner, Andrea Suckro

Assignment 2 - Cats and Dogs
"""

# Task 2 - Setup
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

np.random.seed(1)

###################################################################
# All the functions necessary for the network
###################################################################

actFun = lambda x : 1.7159 * np.tanh(2/3 * x)
actFunDeriv = lambda x : 2/3 * 1.7159 * (1 - np.tanh(2/3 * x)**2)
net = lambda x,w : actFun(actFun(x * w[0]) * w[1])
net_error = lambda y,t : 0.5 * (y - t)**2
eta = lambda c,t: c/t
norm = lambda v: (v - np.mean(v)) / np.std(v)


###################################################################
# Task 4 - Training Data
###################################################################
sampleSize = 30

cats = np.random.normal(25, 5, sampleSize)
cats = np.vstack([cats, np.ones(cats.shape)])

dogs = np.random.normal(45, 15, sampleSize)
dogs = np.vstack([dogs, -1 * np.ones(dogs.shape)])

catdog = np.hstack([cats, dogs])
# normalize the data
catdog[0,:] = norm(catdog[0,:])


###################################################################
# Neural Network with Backprop
###################################################################
w = np.array([1., 1.])
c = 0.3

# training
for t, (sample, target) in enumerate(catdog.T, 1):
     y = net(sample, w)
     w[1] = w[1] - eta(c,t) * (y - target) * actFunDeriv(y) * actFun(sample * w[0])
     w[0] = w[0] - eta(c,t) * (y - target) * actFunDeriv(actFun(w[0] * sample)) * sample

print("Weights after backprop w0 {} and w1 {}".format(w[0], w[1]))

# testing todo: implement rest
# for t, (sample, target) in enumerate(catdog.T, 1):
#     y = net(sample, w)


###################################################################
# Finding the error plane
###################################################################

def test_weight(w):
    error = 0
    for t, (sample, target) in enumerate(catdog.T, 1):
        error += net_error(net(sample, w), target)
    return error / len(catdog.T)

# brute force calculating all combinations
error_results = [(x, y, test_weight(np.array((x, y)))) for x in np.arange(-2, 2, 0.05) for y in np.arange(-2, 2, 0.05)]

# Task 5 - Error function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
unpack = lambda x, y, z : (x, y, z)
x, y, z = unpack(*zip(*error_results))
ax.scatter(x,y,zs=z);

# TODO basti: find out why this looks so shitty and is so slow
#X, Y = np.meshgrid(x, y)
#cp = ax.plot_surface(X, Y, z , cmap = plt.cm.coolwarm)
#plt.colorbar(cp)

ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('error')
plt.show()


