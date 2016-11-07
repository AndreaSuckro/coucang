"""
Assignment of group Sunda slow loridae: Alexander Höreth, Sebastian Höffner, Andrea Suckro

Assignment 3 - MNIST
"""
import tensorflow as tf

import mnist
import matplotlib.pyplot as plt
import numpy as np

# plot some samples of the data
# 1) run the data.py script to download the data from the web

data = mnist.MNIST()

trainingData = data.trainingData
trainingLabels = data.trainingLabels

testData = data.testData
testLabels = data.testLabels

"""
plt.figure('Plot of some examples')
n = 1
for i in np.random.randint(0, len(testData), 9):
    plt.subplot(3,3,n)
    plt.imshow(testData[i], cmap='Greys_r')
    n = n + 1

plt.show()
"""

"""
4 - Investigate the data

I think the only problem would be the difference between German ones and sevens compared to
the ones and sevens in the data set, since they have additional lines. The other numbers should be no problem.
"""
batch_size = 5
max_epochs = 2

##################################################
# 6 - Implement the DFG
input = tf.placeholder(tf.float32, [None, 28*28], name='input')
desired = tf.placeholder(tf.float32, [None, 10], name='target')

weights = tf.Variable(tf.random_normal([28*28, 10]), dtype = "float32", name='weights')
bias = tf.Variable(tf.zeros([10]), dtype = "float32", name='biases')
neuron_out = tf.nn.softmax(tf.add(tf.matmul(input, weights), bias), name='net_output')

# learning
cross_entropy = -tf.reduce_sum(desired * tf.log(neuron_out))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
minimizer = optimizer.minimize(cross_entropy)

rand_idx = np.random.random_integers(0, len(trainingData)-1, size=[batch_size,max_epochs])

# test
correct_prediction = tf.equal(tf.argmax(neuron_out, 1), tf.argmax(desired, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

################################################
# run the model
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    for i in range(max_epochs):
        train_idx = rand_idx[:,i]

        trainBatchData = trainingData[train_idx]
        trainBatchData = np.reshape(trainBatchData, [-1, 28*28])

        trainingBatchLabels = np.zeros([batch_size, 10])

        #trainBatchData = np.random.random_integers(0, len(trainingData) - 1, batch_size)

        numberLabels = trainingLabels[train_idx]
        for idx, numberLabel in enumerate(numberLabels):
            trainingBatchLabels[idx, numberLabel] = 1

        minimizer.run(feed_dict = {input: trainBatchData, desired: trainingBatchLabels})

    # evaluate network
    testFlatData = np.reshape(testData, [-1,28*28])
    testFlatLabels = np.zeros([len(testLabels), 10])
    for idx, numberLabel in enumerate(testLabels):
        testFlatLabels[idx, numberLabel] = 1

    print(accuracy.eval(feed_dict={input: testFlatData, desired: testFlatLabels}))
