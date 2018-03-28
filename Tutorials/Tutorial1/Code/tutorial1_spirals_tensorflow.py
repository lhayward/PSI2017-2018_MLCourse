############ PHYS 777: MACHINE LEARNING FOR MANY-BODY PHYSICS, TUTORIAL 2 ############
### Code by Lauren Hayward Sierens and Juan Carrasquilla
###
### This code builds a simple data set of spirals with K branches and then implements
### and trains a simple feedforward neural network to classify its branches.
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

### Activation function
def sigmoid(x):
  return 1. / (1. + np.exp(-x))

############################################################################
####################### CREATE AND PLOT THE DATA SET #######################
############################################################################

D = 2  # dimensionality of the vectors to be learned
N = 50 # number of points per branch
K = 3  # number of branches
x_train = np.zeros((N*K,D)) # matrix containing the dataset
y_train = np.zeros(N*K, dtype='uint8') # labels

mag_noise = 0.3 #0.2
dTheta    = 5 #4
# data generation
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.01,1,N) # radius
  t = np.linspace(j*(2*np.pi)/K,j*(2*np.pi)/K + dTheta,N) + np.random.randn(N)*mag_noise # theta
  x_train[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
  y_train[ix] = j

### Plot the data set:
fig = plt.figure(figsize=(6,6))
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=40)#, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('x')
plt.ylabel('y')
fig.savefig('spiral_raw.pdf')

############################################################################
##################### DEFINE THE NETWORK ARCHITECTURE ######################
############################################################################

# Computational graph for tensorflow
x = tf.placeholder(tf.float32, [None, D])
yl = tf.placeholder(tf.int32,[None])

n_L = [D,20,K] #size of each layer
num_layers = len(n_L)
W = [ tf.Variable( tf.random_normal([n_L[i], n_L[i+1]], mean=0.0, stddev=0.01, dtype=tf.float32) ) for i in range(num_layers-1) ]
b = [tf.Variable(tf.zeros([n_L[i]])) for i in range(1,num_layers)]
                
a=[None for i in range(num_layers-1)]
a[0] =  tf.nn.sigmoid( tf.matmul(x, W[0]) + b[0] )
for i in range(1,num_layers-1):
  a[i] = tf.nn.sigmoid( tf.matmul(a[i-1], W[i]) + b[i] )

#W1 = tf.Variable(tf.random_normal([ D, h], mean=0.0, stddev=0.01, dtype=tf.float32))
#b1 = tf.Variable(tf.zeros([h]))
#W2 = tf.Variable(tf.random_normal([ h, K], mean=0.0, stddev=0.01, dtype=tf.float32))
#b2 = tf.Variable(tf.zeros([K]))
#
## forward pass
#z1 = tf.matmul(x, W1) + b1
#a1 = tf.nn.sigmoid(z1)
#z2 = tf.matmul(a1, W2) + b2
#a2 = tf.nn.sigmoid(z2)
#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yl,logits=z2)) # tf shortcut
eps=0.0000000001
cross_entropy = tf.reduce_mean(-tf.reduce_sum( tf.one_hot(yl,depth=K) * tf.log(a[-1]+eps) +  (1.0-tf.one_hot(yl,depth=K) )*tf.log(1.0-a[-1] +eps) , reduction_indices=[1])) # a little more explicit

# some hyperparameters
step_size = 1.0 #e-0

train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # gradient descent loop
    num_examples = x_train.shape[0]
    for i in range(10000):
        #
        sess.run(train_step, feed_dict={x: x_train,yl:y_train})
        if i % 1000 == 0:
            loss=sess.run(cross_entropy,feed_dict={x:x_train, yl:y_train})
            print "iteration %d: loss %f" % (i, loss)

    #hidden_layer = np.maximum(0, np.dot(X, W) + b)
    #scores = np.dot(hidden_layer, W2) + b2
    scores_=sess.run(a[-1],feed_dict={x:x_train, yl:y_train})
    predicted_class = np.argmax(scores_, axis=1)
    print 'training accuracy: %.2f' % (np.mean(predicted_class == y_train))
    
    # getting the trained weights for plotting the classifier
    #W1=sess.run(W1)
    #W2=sess.run(W2)
    #b1=sess.run(b1)
    #b2=sess.run(b2)
    

    # plot the resulting classifier
    d_xy = 0.02
    x_min, x_max = x_train[:, 0].min() - 0.25, x_train[:, 0].max() + 0.25
    y_min, y_max = x_train[:, 1].min() - 0.25, x_train[:, 1].max() + 0.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, d_xy),
                         np.arange(y_min, y_max, d_xy))
    #Z = np.dot(sigmoid( np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2
    Z = sess.run(a[-1],feed_dict={x:np.c_[xx.ravel(), yy.ravel()]})
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z, K, alpha=0.8)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=40)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('x')
plt.ylabel('y')
fig.savefig('spiral_net_results.pdf')

plt.show()
