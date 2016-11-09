#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import choice, random_integers
from mnist import MNIST
import tensorflow as tf


BATCHSIZE = 100
EPOCHS = 10000
TESTSTEPSIZE = 100


data = MNIST()


###############################################################################
# Visualize Data
plt.ioff()
fig = plt.figure('MNIST Data')
for i, sample in enumerate(choice(len(data.trainingData), 9, replace=False)):
    sbplt = plt.subplot(331+i)
    sbplt.axis('off')
    sbplt.set_title(data.trainingLabels[sample])
    sbplt.imshow(data.trainingData[sample], cmap='gray')
plt.show()


###############################################################################
# Model
x = tf.placeholder(tf.float32, (None, 28, 28), name='input')
t = tf.placeholder(tf.int64, (None), name='target')

W = tf.Variable(tf.random_normal((784, 10)), name='weights')
b = tf.Variable(tf.zeros((10,)), name='biases')
y = tf.matmul(tf.reshape(x, (-1, 28*28)), W) + b

entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, t, name='entropy')
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), t)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###############################################################################
# Training
sess = tf.Session()
sess.run(tf.initialize_all_variables())

plt.ion()
plt.axis([0, EPOCHS, 0, 1])
plt.title('Training...')
acc = []

for i in range(EPOCHS):
    batch_x, batch_t = data.getTrainingBatch(BATCHSIZE)
    sess.run(train_step, {x: batch_x, t: batch_t})

    if (i % TESTSTEPSIZE == 0):
        acc.append(sess.run(accuracy, {x: data.testData, t: data.testLabels}))
        plt.plot(range(0, i+1, TESTSTEPSIZE), acc, 'r-')
        plt.ylabel('Accuracy {:1.3f}'.format(acc[-1]))
        plt.xlabel('Epoch {:d}'.format(i))
        plt.pause(0.001)

plt.ioff()
plt.title('Training done.')
plt.show()
