import tensorflow as tf 
import numpy as np 

trX = np.linspace(-1, 1, 101)   # an array of 101 numbers evenly spaced from -1 to 1
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear (2x) but with some random noise
                # random n's in the shape of trX

X = tf.placeholder("float")
Y = tf.placeholder("float")


def model(X, w):
    return tf.mul(X, w)     # lr is just X*w so this model line is pretty simple

w = tf.Variable(0.0, name="weights")    # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)

cost = (tf.pow(Y-y_model, 2))   # (Y-y_model)^2 -> SSE

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(100):
    for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})

print(sess.run(w))      # should give a weight of approx. 2 (our linear transformation)
