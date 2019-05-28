'''
difference is that logistic regression is used to classify things rather than
try to predict an outcome
similar to linear regression with the variables having different meaning
formula is the same y=W.x+b
y is now a class of what we want to predict(in this case 1-9) no longer can be the scalar of the matrix, as that isn't a class we want to classify
x is still the features, in this case every pixel in the image
cost is the aggregate weather the prddiction correct or wrong based on the actual
outcome and training goals are the same

in this case the machine will come up with a prediction for how likely it will be that each class if the right class(in a 1X#ofclasses matrix),
the highest value is the assumed prediction, the prediction value is a sum from all the pixels

cost function changes using crossEntroby now
convert actual image in to a probablity vector, predicted image into a probability vector
use crossentropy to find the difference


use one hot vector for actual images so that every class if 0 except the corresponding element is 1
cross entropy used to compare the similarity between 2 graphs
cross entropy requires that the sum of the vector adds to one, one hot vector satisfies this req
use softmax to convt prediction graph into a one hot vector

by taking the -log(softmax(y_i)) we create a gradient decent so we can apply linear regression tools
'''
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)
# Step 2: Define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 25
# Step 3: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9.
# each label is one hot vector.
X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])
# Step 4: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")
# Step 5: predict Y from X and w, b
# the model that returns probability distribution of possible label of the image
# through the softmax layer
# a batch_size x 10 tensor that represents the possibility of the digits
logits = tf.matmul(X, w) + b
# Step 6: define loss function
# use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
loss = tf.reduce_mean(entropy) # computes the mean over examples in the batch
# Step 7: define training op
# using gradient descent with learning rate of 0.01 to minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	n_batches = int(MNIST.train.num_examples/batch_size)
	for i in range(n_epochs): # train the model n_epochs times
		for _ in range(n_batches):
			X_batch, Y_batch = MNIST.train.next_batch(batch_size)
			sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
# average loss should be around 0.35 after 25 epochs
# test the model
		n_batches = int(MNIST.test.num_examples/batch_size)
		total_correct_preds = 0
		for i in range(n_batches):
			X_batch, Y_batch = MNIST.test.next_batch(batch_size)
			_, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
			feed_dict={X: X_batch, Y:Y_batch})
			preds = tf.nn.softmax(logits_batch)
			correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
			accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # similar
			total_correct_preds += sess.run(accuracy)
		print ("Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples))