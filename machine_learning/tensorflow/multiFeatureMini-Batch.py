'''
introduce a second feature rooms
now instead of predicting a line we are trying to predict a plane for the 
3d graph. 

y=W.x +W2.x2+b new model with new features

as more and more features are made, can't keep making placeholder variables 
to keep track of features. Use matrices instead.

model with n features:
y=x1.W1+x2.W2+...xn.Wn+b
need different placeholders and weights for each feature
can be represented by a matrix

feature matrix vs weight matrix
1 r X n c  matmul n r X 1 c  results in 1 r X 1 c matrix

{x1 x2 ... xn} x {W!
				  W2...
				  WN }
can be created by
x=tf.placeholder(tf.float32,[1,n])
W =tf.Variable(tf.zeros[n,1])
in case of multiple datapoints being used
x=tf.placeholder(tf.float32,[m,n])
W =tf.Variable(tf.zeros[n,1])
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# CUSTOMIZABLE: Collect/Prepare data
datapoint_size = 1000
batch_size = 1000
steps = 10000
actual_W1 = 10
actual_W2 = 20
actual_W3 = 4
actual_b = 5
learn_rate = 0.001



x = tf.placeholder(tf.float32, [None, 3], name="x")
W = tf.Variable(tf.zeros([3,1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")
with tf.name_scope("Wx_b") as scope:
  product = tf.matmul(x,W)
  y = product + b


y_ = tf.placeholder(tf.float32, [None, 1])

# Cost function sum((y_-y)**2)
with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean(tf.square(y_-y))

# Training using Gradient Descent to minimize cost
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

all_xs = []
all_ys = []
update_w1 = []
update_w2 = []
update_w3 = []
update_b = []
update_iterations = []
cost_arr=[]
arr_actual_w1 = []
arr_actual_w2 = []
arr_actual_w3 = []
arr_actual_b = []
for i in range(datapoint_size):
  x_1 = i%10
  x_2 = np.random.randint(datapoint_size/2)%10
  x_3 = i*35%10
  y = actual_W1 * x_1 + actual_W2 * x_2 +actual_W3 * x_3+ actual_b
  # Create fake data for y = W.x + b where W = [2, 5], b = 7
  all_xs.append([x_1, x_2, x_3])
  all_ys.append(y)

all_xs = np.array(all_xs)
all_ys = np.transpose([all_ys])

sess = tf.Session()


init = tf.global_variables_initializer()
sess.run(init)

for i in range(steps):
  if datapoint_size == batch_size:
    batch_start_idx = 0
  elif datapoint_size < batch_size:
    raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
  else:
    batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
  batch_end_idx = batch_start_idx + batch_size
  batch_xs = all_xs[batch_start_idx:batch_end_idx]
  batch_ys = all_ys[batch_start_idx:batch_end_idx]
  xs = np.array(batch_xs)
  ys = np.array(batch_ys)
  all_feed = { x: all_xs, y_: all_ys }
  feed = { x: xs, y_: ys }
  sess.run(train_step, feed_dict=feed) #use all_feed for multiFeatureBatch
  print("After %d iteration:" % i)
  print("W: %s" % sess.run(W))
  print("b: %f" % sess.run(b))
  print("cost: %f" % sess.run(cost, feed_dict=all_feed))
  if(i%100 == 0):
  	update_w1.append(sess.run(W)[0])
  	update_w2.append(sess.run(W)[1])
  	update_w3.append(sess.run(W)[2])
  	update_b.append(sess.run(b))
  	cost_arr.append(sess.run(cost, feed_dict=all_feed))
  	arr_actual_b.append(actual_b)
  	arr_actual_w1.append(actual_W1)
  	arr_actual_w3.append(actual_W3)
  	arr_actual_w2.append(actual_W2)
  	update_iterations.append(i)
  

update_iterations=np.array(update_iterations)
update_b=np.array(update_b)
update_w1=np.array(update_w1)
update_w2=np.array(update_w2)
arr_actual_w1=np.array(arr_actual_w1)
arr_actual_w2=np.array(arr_actual_w2)
arr_actual_b=np.array(arr_actual_b)
cost_arr = np.array(cost_arr)



plt.subplot(5,1,1)
plt.title("3 feature mini-batch linear regression")
plt.plot(update_iterations,update_w1)
plt.plot(update_iterations,arr_actual_w1,color="red")
plt.ylabel("w1")
plt.xlabel("iteration")
plt.ylim(0,actual_W1*1.3)

plt.subplot(5,1,2)
plt.plot(update_iterations,update_w2)
plt.plot(update_iterations,arr_actual_w2,color="red")
plt.ylabel("w2")
plt.xlabel("iteration")
plt.ylim(0,actual_W2*1.3)

plt.subplot(5,1,3)
plt.plot(update_iterations,update_w3)
plt.plot(update_iterations,arr_actual_w3,color="red")
plt.ylabel("w3")
plt.xlabel("iteration")
plt.ylim(0,actual_W2*1.3)


plt.subplot(5,1,4)
plt.plot(update_iterations,update_b)
plt.plot(update_iterations,arr_actual_b,color="red")
plt.ylabel("b")
plt.xlabel("iteration")
plt.ylim(0,actual_b*1.3)


plt.subplot(5,1,5)
plt.plot(update_iterations,cost_arr)
plt.ylabel("cost")
plt.xlabel("iteration")
plt.ylim(0,7)

plt.show()