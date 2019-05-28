#refer to logistic regression notes
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import keyboard


mnist = input_data.read_data_sets("tmp/data",one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 600
n_nodes_hl3 = 700

n_classes = 10
batch_size = 50
learning_rate = .001

x=tf.placeholder(tf.float32,[None,28*28])
y=tf.placeholder(tf.float32)

def neural_network_model(data):
	hidden_1_layer={"weights":tf.Variable(tf.random_normal([784,n_nodes_hl1])),
					"biases":tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer={"weights":tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					"biases":tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer={"weights":tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					"biases":tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer={"weights":tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
					"biases":tf.Variable(tf.random_normal([n_classes]))}

	l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
	l1=tf.nn.relu(l1)

	l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
	l2=tf.nn.relu(l2)

	l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
	l3=tf.nn.relu(l3)

	output=tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])

	return output
def use_neural_network(input_data):
    prediction = neural_network_model(x)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,'./mnist/test_deep_net.ckpt')
        
        input_data = np.array(input_data)
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:input_data}),1)))
        print(result[0])



def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	#tf.summary.histogram("cost",cost)
	#tf.summary.histogram("prediction",prediction)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	hm_epochs = 10
	saver=tf.train.Saver()

	with tf.Session() as sess:
		#train_writer = tf.summary.FileWriter('./logs/1/train',sess.graph)
		sess.run(tf.global_variables_initializer())
		#saver.restore(sess,'./mnist/test_deep_net.ckpt')
		
		cnt = 0
		for epoch in range(hm_epochs):
			epoch_cost=0
			for blah1 in range(int(mnist.train.num_examples/batch_size)):
				cnt+=1
				#merge = tf.summary.merge_all()
				batch_x,batch_y=mnist.train.next_batch(batch_size)
				#summary, blah2,c = sess.run([merge,optimizer,cost],feed_dict = {x: batch_x, y: batch_y})
				blah2,c = sess.run([optimizer,cost],feed_dict = {x: batch_x, y: batch_y})
				#train_writer.add_summary(summary,cnt)
				epoch_cost += c
			print("Epoch: ", epoch)
			print("loss: ", epoch_cost)
			correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
			print("Accuracy: ",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
		#tf.summary.historgram("Accuracy", accuracy)
		saver.save(sess,'./mnist/test_deep_net.ckpt')
#train_neural_network(x)
batch_x,batch_y = mnist.train.next_batch(1)
data = batch_x[0].reshape(28,28)
plt.imshow(batch_x[0].reshape(28,28))
use_neural_network(batch_x)
plt.show(block=True)

















