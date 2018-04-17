############ PHYS 777: MACHINE LEARNING FOR MANY-BODY PHYSICS, TUTORIAL 2 ############
### Dataset and code by Lauren Hayward Sierens and Juan Carrasquilla
###
### This code will classify the phases of the classical Ising gauge theory.
### The classification does not use machine learning methods and is based on
### knowledge of the Hamiltonian and the topological Wilson loop.
######################################################################################

import numpy as np

### Read in the spin configurations: ###
fileName  = 'gaugeTheoryConfigs.txt'          # The file where the configurations are stored
configs   = np.loadtxt(fileName,dtype='int8') # Read the data from file

N_configs = configs.shape[0]                  # Total number of configurations given
N_spins   = configs.shape[1]                  # Total number of spins per configuration
N_sites   = N_spins/2                         # Total number of lattice sites
L         = int(np.sqrt(N_sites))             # Linear size of the lattice
J         = 1                                 # Coupling parameter

### Loop over all configurations: ###
for c in range(N_configs):
    x = configs[c]  # A numpy array of length N_spins that store a spin configuration
