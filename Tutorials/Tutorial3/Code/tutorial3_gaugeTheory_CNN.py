############ PHYS 777: MACHINE LEARNING FOR MANY-BODY PHYSICS, TUTORIAL 3 ############
### Dataset and code by Lauren Hayward Sierens and Juan Carrasquilla
###
### This code will classify the phases of the classical Ising gauge theory
### using a neural network with a convolutional layer.
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

### Data parameters: ###
num_labels  = 2     # Number of labels (T=0 and T=infinity here)
num_sublattices = 2 # Number of sublattices for the gauge theory lattice

### Hyperparameters: ###
patch_size     = 2      # Size of the filters
num_filters    = 64     # Number of output channels in the convolutional layer
stride_xy      = 1      # The size of the jumps to take as one slides the filters across the image (the lattice length L should be divisible by stride_xy)
nH2            = 64     # Number of hidden neurons in the fully-connected layer
dropout_prob   = 0.5    # Probability of keeping neurons in the dropout layer
learning_rate  = 0.0001 # Learning rate for training algorithm
minibatch_size = 2000   # Mini-batch size (N_train needs to be divisible by minibatch_size)

### Other parameters: ###
N_epochs = 50           # Number of times to iterate through all of the data

seed=1
np.random.seed(seed)
tf.set_random_seed(seed)

### Plotting parameters: ###
#Specify font sizes for plots:
plt.rcParams['axes.labelsize']  = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

plt.ion() # turn on interactive plotting mode

############################################################################
########################### READ IN THE DATA SET ###########################
############################################################################

### Read in the training data: ###
x_train_orig = np.loadtxt( 'Data_Problem1/x_train.txt', dtype='uint8' )
y_train      = np.loadtxt( 'Data_Problem1/y_train.txt', dtype='uint8' )
N_train      = x_train_orig.shape[0]
N_spins      = x_train_orig.shape[1]
L            = int( np.sqrt(N_spins/2) )

### Read in the test data: ###
x_test_orig = np.loadtxt( 'Data_Problem1/x_test.txt', dtype='uint8' )
y_test      = np.loadtxt( 'Data_Problem1/y_test.txt', dtype='uint8' )
N_test      = x_test_orig.shape[0]

### Enlarge the datapoints based on the patch size (because of periodic boundary conditions): ###
L_enlarged = L+patch_size-1
n0 = 2*(L_enlarged)**2
def enlarge_data(N_samples,data_orig):
    data_enlarged = np.zeros((N_samples,n0))

    for iy in range(L):
        data_enlarged[:,2*iy*L_enlarged:(2*iy*L_enlarged + 2*L)] = data_orig[:,2*iy*L:2*(iy+1)*L]
        data_enlarged[:,(2*iy*L_enlarged + 2*L):2*(iy+1)*L_enlarged] = data_orig[:,2*iy*L:(2*iy*L+2*(patch_size-1))]
    data_enlarged[:,2*L*L_enlarged:] = data_enlarged[:,0:2*L_enlarged*(patch_size-1)]
    return data_enlarged

x_train = enlarge_data(N_train, x_train_orig)
x_test  = enlarge_data(N_test,  x_test_orig)

print("N_train    = %d\nN_test     = %d\nL          = %d\nL_enlarged = %d\n" %(N_train,N_test,L,L_enlarged))

############################################################################
##################### DEFINE THE NETWORK ARCHITECTURE ######################
############################################################################

x = tf.placeholder(tf.float32, shape=[None, n0]) # Placeholder for the spin configurations
x_reshaped = tf.reshape( x, [-1,L_enlarged,L_enlarged,num_sublattices] )
y = tf.placeholder(tf.int32, shape=[None]) # Labels

### Layer 1 (Convolutional layer): ###
W1 = tf.Variable( tf.truncated_normal([patch_size, patch_size, num_sublattices, num_filters], mean=0.0, stddev=0.01, dtype=tf.float32) )
b1 = tf.Variable( tf.constant(0.1,shape=[num_filters]) )

# Apply the convolution (note that 'VALID' means no padding):
z1 = tf.nn.conv2d(x_reshaped, W1, strides=[1, stride_xy, stride_xy, 1], padding='VALID') + b1
a1 = tf.nn.relu( z1 )
n_a1 = num_filters*(L/stride_xy)**2 # Number of outputs in the vector a1
a1_flattened = tf.reshape( a1, [-1,n_a1])

### Layer 2 (Fully-connected layer): ###
W2 = tf.Variable( tf.truncated_normal([n_a1,nH2], mean=0.0, stddev=0.01, dtype=tf.float32) )
b2 = tf.Variable( tf.constant(0.1,shape=[nH2]) )
z2 = tf.matmul(a1_flattened, W2) + b2
a2 = tf.nn.relu( z2 )

# Dropout: To reduce overfitting, we apply dropout to the neurons a2 (before the final output layer).
# We create a placeholder for the probability that a neuron's output is kept during dropout, which
# allows us to turn dropout on during training, and turn it off during testing. TensorFlow's
# tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so
# dropout works without any additional scaling.
keep_prob = tf.placeholder("float")
a2_drop = tf.nn.dropout(a2, keep_prob)

### Layer 3 (Fully-connected layer): ###
W3 = tf.Variable( tf.truncated_normal([nH2,num_labels], mean=0.0, stddev=0.01, dtype=tf.float32) )
b3 = tf.Variable( tf.constant(0.1,shape=[num_labels]) )
z3 = tf.matmul(a2_drop, W3) + b3
a3 = tf.nn.softmax( z3 )

### Network output: ###
aL = a3

### Cost function: ###
y_onehot = tf.one_hot(y,depth=num_labels) # labels are converted to one-hot representation
eps=0.0000000001 # to prevent the logs from diverging
cross_entropy = tf.reduce_mean(-tf.reduce_sum( y_onehot * tf.log(aL+eps) +  (1.0-y_onehot)*tf.log(1.0-aL+eps) , reduction_indices=[1]))
cost_func = cross_entropy

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_func)

##############################################################################
################################## TRAINING ##################################
##############################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

iteration_list = []
cost_training  = []
acc_test       = []

### Train using mini-batches for several epochs: ###
permut = np.arange(N_train)
num_iterations = 0
for epoch in range(N_epochs):
    np.random.shuffle(permut) # Randomly shuffle the indices
    x_shuffled = x_train[permut,:]
    y_shuffled = y_train[permut]

    #Loop over all the mini-batches:
    for b in range(0, N_train, minibatch_size):
        x_batch = x_shuffled[b:b+minibatch_size,:]
        y_batch = y_shuffled[b:b+minibatch_size]
        sess.run(train_step, feed_dict={x: x_batch, y:y_batch, keep_prob:dropout_prob})
        num_iterations = num_iterations + 1

        #Print results every 10 iterations of the training algorithm:
        if num_iterations % 10 == 0:
            cost_train = sess.run(cost_func,feed_dict={x:x_train, y:y_train, keep_prob:1.0})

            test_output = sess.run(aL,feed_dict={x:x_test, y:y_test, keep_prob:1.0})
            predicted_class = np.argmax(test_output, axis=1)
            accuracy_test = np.mean(predicted_class == y_test)

            print( "Iteration %d:\n  Training cost %f\n  Test accuracy %f\n" % (num_iterations, cost_train, accuracy_test) )

            iteration_list.append(num_iterations)
            cost_training.append(cost_train)
            acc_test.append(accuracy_test)

            fig = plt.figure(1,figsize=(10,4))
            fig.subplots_adjust(hspace=.3,wspace=.3)
            plt.clf()
            
            ### Plot the cost function during training: ###
            plt.subplot(121)
            plt.plot(iteration_list,cost_training,'o-')
            plt.xlabel('Iteration')
            plt.ylabel('Training cost')
            
            ### Plot the test accuracy: ###
            plt.subplot(122)
            plt.plot(iteration_list,acc_test,'o-')
            plt.xlabel('Iteration')
            plt.ylabel('Test accuracy')
            plt.pause(0.1)

plt.savefig('gaugeTheoryClassification_CNN_results.pdf') # Save the figure showing the results in the current directory
