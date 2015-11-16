import tensorflow as tf 
import numpy as np 
import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w):
    return tf.matmul(X, w)      #   same model as linear reg because cost function

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY = mnist.train.images, mnist.train.labels
teX, teY = mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10])     #   like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

#   compute mean cross entropy ( softmax is just multinomial linear regression ) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
        #   http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
        #   http://tensorflow.org/api_docs/python/nn.md#softmax_cross_entropy_with_logits

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
                
            #   argmax(input, dimension)
predict_op = tf.argmax(py_x, 1 )

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#   100 epochs
for i in range(100):

    #   trains the whole dataset in batches of 128
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

    #   what percentage of the things are what we think they are?
    print i, np.mean(np.argmax(teY, axis=1) ==
                    sess.run(predict_op, feed_dict={X: teX, Y: teY}))

