"""
Solution of the Group Sunda Slow Loridae

Task 4 - Convolutional Neural Networks
"""
import tensorflow as tf
import numpy as np
from mnist import MNIST

#################################################################
# Load Data

data = MNIST()

# add magic single dimension to data
testLabel = data.testLabels
testData = data.testData[:,:,:,np.newaxis]

trainingLabel = data.trainingLabels
trainingData = data.trainingData[:,:,:,np.newaxis]


def getTrainingBatch(n=100):
    sample = np.random.random_integers(0, len(trainingData) - 1, n)
    return trainingData[sample], trainingLabel[sample]

#################################################################
# Define Network

# placeholders
input = tf.placeholder(tf.float32, (None, 28, 28, 1), name="input")
target = tf.placeholder(tf.int64, (None), name="target")

# variable definition and network structure
# first layer
# convolution of the filter with the image
w1 = tf.Variable(tf.random_normal((5, 5, 1, 32)), name="w1") # create kernels
b1 = tf.Variable(tf.constant(0.1, shape=[32])) # create biases
y1 = tf.tanh(tf.nn.conv2d(input, w1, strides = [1, 1, 1, 1], padding = 'SAME', name='conv_layer') + b1) # output of the first layer is the convolution of the kernels + the bias in the activation function

pool1 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding= 'SAME', name='pool1')

# second layer
# and now we want to do max pooling on this
w2 = tf.Variable(tf.random_normal((5, 5, 32, 64)), name="w2")
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
y2 = tf.tanh(tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding= "SAME", name= 'conv_layer_2') + b2)

pool2 = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool2')

# third layer
w3 = tf.Variable(tf.random_normal((64*7*7, 1024)), name='weights')
b3 = tf.Variable(tf.constant(0.1, shape=[1024]))
y3 = tf.nn.relu_layer(tf.reshape(pool2,(-1, 64*7*7)), w3, b3, name="first_ffn_layer")

# fourth layer
w4 = tf.Variable(tf.random_normal((1024, 10)), name="last_ffn_layer")
b4 = tf.Variable(tf.constant(0.1, shape=[10]))
y4 = tf.matmul(y3, w4) + b4

# define the optimizers
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y4, target, name='entropy')
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(y4, 1), target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#################################################################
# Train Network
EPOCHS = 1000
BATCH = 300

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1, EPOCHS+1):
    batch_x, batch_t = getTrainingBatch(BATCH)
    sess.run(train_step, {input: batch_x, target: batch_t})

print(sess.run(accuracy, {input: data.testData, target: data.testLabels}))