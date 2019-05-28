'''
linear model goal to  minimize the squared(avoid neg) of the difference
between the actual value and the predicted value created by our model.
(call diff)

(y=W.x+b) function x is house size y is house price, change W to figure out
how to minimize difference. W also referred to as gradient tweak b to change 
the starting position of the model. By testing different values of W and b 
we can find the ideal numbers for the best fit line.

use gradient decent to try and figure out how to change W and b. This is 
called training. think of a slightly decreasing function that has a peak
of decent, then after that peak it increases again.(like binary search)
'''
import tensorflow as tf
import numpy as np

x=tf.placeholder(tf.float32,[None,1]) #x is input so the 1 is # of features
W=tf.Variable(tf.zeros([1,1]))#first arguemnt # of output, second argument # of features
b=tf.Variable(tf.zeros([1]))# only argument is # of features
m=tf.matmul(x,W)+b #multiply matrixs to get "slope"
y_=tf.placeholder(tf.float32,[None,1])#since y is output # of outputs
diff=tf.reduce_sum(tf.pow((y_-m),2)) #takes the sum of each point from the line


init = tf.global_variables_initializer() #clear variables
train_step=tf.train.GradientDescentOptimizer(.00001).minimize(diff) #.00001 is train rate, minimizing diff which is defined above(the trendline)
sess=tf.Session() #create session
sess.run(init) #start session
#create fake data with slope of 2(just for testing)
for i in range(100):
	xs=np.array([[i]]) #creating data while analyzing it, just for simplicity
	ys=np.array([[2*i]])
	# train
	data = {x:xs,y_:ys} #setup the fake data in a form usable by tf
	sess.run(train_step,feed_dict=data) #give the session the data
	print("Iteration: ",i) 
	print("W: ",sess.run(W))
	print("b: ",sess.run(b))

