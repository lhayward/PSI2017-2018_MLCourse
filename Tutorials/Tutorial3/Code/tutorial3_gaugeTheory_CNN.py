############ PHYS 777: MACHINE LEARNING FOR MANY-BODY PHYSICS, TUTORIAL 3 ############
### Dataset and code by Lauren Hayward Sierens and Juan Carrasquilla
###
######################################################################################

import numpy as np
import tensorflow as tf

### Read in the training data: ###
x_train_orig = np.loadtxt( 'x_train.txt', dtype='uint8' )
y_train      = np.loadtxt( 'y_train.txt', dtype='uint8' )
N_train      = x_train_orig.shape[0]
N_spins      = x_train_orig.shape[1]
L            = int( np.sqrt(N_spins/2) )

### Read in the test data: ###
x_test_orig = np.loadtxt( 'x_test.txt', dtype='uint8' )
y_test      = np.loadtxt( 'y_test.txt', dtype='uint8' )
N_test      = x_test_orig.shape[0]

#N_train=2
#L = 4
#xxx = np.zeros((N_train,2*L*L))
#for i in range(N_train):
#    xxx[i] = np.arange(2*L*L)

### Hyperparameters: ###
patch_size = 3
num_filters = 64  # Number of output channels

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

x = tf.placeholder(tf.float32, shape=[None, n0]) # placeholder for the spin configurations
y = tf.placeholder(tf.int32, shape=[None]) # labels
