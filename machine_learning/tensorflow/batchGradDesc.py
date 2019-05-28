'''
Uses all the data at once, lowers number of epochs increases computational resources


can incorporate a learn rate that decreases over time as we assume we get closer to the optimal
gradient descent
'''
import numpy as np
import tensorflow as tf

# CUSTOMIZABLE: Collect/Prepare data
datapoint_size = 1000
batch_size = 1
steps = 1000
actual_W = 2
actual_b = 10
initial_learn_rate = 0.01

# Model linear regression y = Wx + b
x = tf.placeholder(tf.float32, [None, 1], name="x")
W = tf.Variable(tf.zeros([1,1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")
learn_rate = tf.placeholder(tf.float32,shape=[])
with tf.name_scope("Wx_b") as scope:
  product = tf.matmul(x,W)
  y = product + b
y_ = tf.placeholder(tf.float32, [None, 1], name="y_")

# Cost function sum((y_-y)**2)
with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean(tf.square(y_-y))
# Training using Gradient Descent to minimize cost
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

all_xs = []
all_ys = []
for i in range(datapoint_size):
  # Create fake data for y = W.x + b where W = 2, b = actual_b
  all_xs.append(i%10)
  all_ys.append(actual_W*(i%10)+actual_b)

all_xs = np.transpose([all_xs])
all_ys = np.transpose([all_ys])

sess = tf.Session()


init = tf.initialize_all_variables()
sess.run(init)

for i in range(steps):
  feed = { x: all_xs, y_: all_ys,learn_rate: initial_learn_rate} #ONLY DIFFERENCE
  sess.run(train_step, feed_dict=feed)
  #print("y: %s" % sess.run(y, feed_dict=feed))
  print("cost: %f" % sess.run(cost, feed_dict=feed))
  print("After %d iteration:" % i)
  print("W: %f" % sess.run(W))
  print("b: %f" % sess.run(b))