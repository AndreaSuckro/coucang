from matplotlib import pyplot as plt
import numpy as np
from numpy.random import choice, random_integers
from mnist import MNIST
import tensorflow as tf


BATCHSIZE = 100
EPOCHS = 10000


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
x = tf.placeholder(tf.float32, (None, 784), name='input')
t = tf.placeholder(tf.int64, (None), name='target')

W = tf.Variable(tf.random_normal((784, 10)), name='weights')
b = tf.Variable(tf.zeros((10,)), name='biases')
y = tf.matmul(x, W) + b

entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, t, name='entropy')
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(entropy)


###############################################################################
# Training & Predictions
correct_prediction = tf.equal(tf.argmax(y, 1), t)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(EPOCHS):
    sample = random_integers(0, len(data.trainingData)-1, BATCHSIZE)
    sess.run(train_step, {
        x: data.trainingData[sample].reshape(-1, 28*28),
        t: data.trainingLabels[sample],
    })

print(sess.run(accuracy, {
    x: data.testData.reshape(-1, 28*28),
    t: data.testLabels,
}))
