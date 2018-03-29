############ PHYS 777: MACHINE LEARNING FOR MANY-BODY PHYSICS, TUTORIAL 2 ############
### Code by Lauren Hayward Sierens and Juan Carrasquilla
###
### This code builds a simple data set of spirals with K branches and then implements
### and trains a simple feedforward neural network to classify its branches.
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(123)
tf.set_random_seed(123)

### Activation function
def sigmoid(x):
  return 1. / (1. + np.exp(-x))

############################################################################
####################### CREATE AND PLOT THE DATA SET #######################
############################################################################

D = 2  # dimensionality of the vectors to be learned
N = 50 # number of points per branch
K = 3  # number of branches

N_train = N*K
x_train = np.zeros((N_train,D)) # matrix containing the dataset
y_train = np.zeros(N_train, dtype='uint8') # labels

mag_noise = 0.4 #0.2
dTheta    = 3 #4
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
fig.savefig('spiral_data.pdf')

############################################################################
##################### DEFINE THE NETWORK ARCHITECTURE ######################
############################################################################

# Computational graph for tensorflow
x = tf.placeholder(tf.float32, [None, D])
yl = tf.placeholder(tf.int32,[None])

n_L = [D,10,K] #size of each layer
num_layers = len(n_L)
W = [ tf.Variable( tf.random_normal([n_L[i], n_L[i+1]], mean=0.0, stddev=0.01, dtype=tf.float32) ) for i in range(num_layers-1) ]
b = [tf.Variable(tf.zeros([n_L[i]])) for i in range(1,num_layers)]
                
a=[None for i in range(num_layers-1)]
a[0] =  tf.nn.sigmoid( tf.matmul(x, W[0]) + b[0] )
for i in range(1,num_layers-1):
  a[i] = tf.nn.sigmoid( tf.matmul(a[i-1], W[i]) + b[i] )

#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yl,logits=z2)) # tf shortcut
eps=0.0000000001
cross_entropy = tf.reduce_mean(-tf.reduce_sum( tf.one_hot(yl,depth=K) * tf.log(a[-1]+eps) +  (1.0-tf.one_hot(yl,depth=K) )*tf.log(1.0-a[-1] +eps) , reduction_indices=[1])) # a little more explicit
regularizer = tf.nn.l2_loss(W[0])
for i in range(1,num_layers-1):
    regularizer = regularizer + tf.nn.l2_loss(W[i])
loss_func = tf.reduce_mean(cross_entropy + 0.001 * regularizer)

step_size = 1.0 #hyperparameter
train_step = tf.train.GradientDescentOptimizer(step_size).minimize(loss_func)

N_epochs = 10000
minibatch_size = 200 #N_train needs to be divisible by batch_size
permut = np.arange(N_train)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # gradient descent loop
    num_examples = x_train.shape[0]
    for i in range(N_epochs):
        np.random.shuffle(permut)
        x_shuffle  = x_train[permut,:]
        y_shuffle = y_train[permut]
        
        for j in range(0, N_train, minibatch_size):
            x_batch = x_shuffle[j:j+minibatch_size,:]
            y_batch = y_shuffle[j:j+minibatch_size]
            sess.run(train_step, feed_dict={x: x_batch,yl:y_batch})
        
        if i % 1000 == 0:
            loss=sess.run(loss_func,feed_dict={x:x_train, yl:y_train})
            print "iteration %d: loss %f" % (i, loss)

    scores_=sess.run(a[-1],feed_dict={x:x_train, yl:y_train})
    predicted_class = np.argmax(scores_, axis=1)
    print 'training accuracy: %.2f' % (np.mean(predicted_class == y_train))

    # plot the resulting classifier
    padding_xy = 0.1
    spacing_xy = 0.02
    x_min, x_max = x_train[:, 0].min() - padding_xy, x_train[:, 0].max() + padding_xy
    y_min, y_max = x_train[:, 1].min() - padding_xy, x_train[:, 1].max() + padding_xy
    xx, yy = np.meshgrid(np.arange(x_min, x_max, spacing_xy),
                         np.arange(y_min, y_max, spacing_xy))
    Z = sess.run(a[-1],feed_dict={x:np.c_[xx.ravel(), yy.ravel()]})
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z, K, alpha=0.8)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=40)

#for j in range(K):
#    r_noNoise = np.linspace(0.01,1, 100 ) # radius
#    t_noNoise = np.linspace(j*(2*np.pi)/K - dTheta/2.0,j*(2*np.pi)/K + dTheta/2.0, 100) # theta
#    plt.plot( r_noNoise*np.cos(t_noNoise), r_noNoise*np.sin(t_noNoise), 'k-' )

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('x')
plt.ylabel('y')
fig.savefig('spiral_net_results.pdf')

plt.show()
