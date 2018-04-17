############ PHYS 777: MACHINE LEARNING FOR MANY-BODY PHYSICS, TUTORIAL 2 ############
### Dataset and code by Lauren Hayward Sierens and Juan Carrasquilla
###
### This code classifies the phases of the classical Ising gauge theory.
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

######################################################################################
############################      SOLUTION TO PART A      ############################
############################   Classify based on energy   ############################
######################################################################################

### Store each lattice site's four nearest neighbours in a neighbours array (using periodic boundary conditions): ###
neighbours = np.zeros((N_sites,4),dtype=np.int)
for i in range(N_sites):
  #neighbour to the right:
  neighbours[i,0]=i+1
  if i%L==(L-1):
    neighbours[i,0]=i+1-L
  
  #upwards neighbour:
  neighbours[i,1]=i+L
  if i >= (N_sites-L):
    neighbours[i,1]=i+L-N_sites
  
  #neighbour to the left:
  neighbours[i,2]=i-1
  if i%L==0:
    neighbours[i,2]=i-1+L
  
  #downwards neighbour:
  neighbours[i,3]=i-L
  if i <= (L-1):
    neighbours[i,3]=i-L+N_sites
#end of for loop

### Function to calculate the total energy: ###
def getEnergy(spins):
  currEnergy = 0
  for i in range(N_sites):
    currEnergy += -J*getPlaquetteProduct(spins,i)
  return currEnergy
#end of getEnergy() function

### Function to calculate the product of spins on plaquette i: ###
def getPlaquetteProduct(spins,i):
  return spins[2*i]*spins[(2*i)+1]*spins[2*neighbours[i,1]]*spins[(2*neighbours[i,0])+1]

num_T0_E=0
### Loop over all configurations: ###
for c in range(N_configs):
    if c%1000==0:
        print('%d configs. checked' %c)
    E = getEnergy(configs[c])
    if E == (-N_sites*J):
        num_T0_E = num_T0_E + 1
print( 'Number topologically ordered (from energy) = %d\n' %num_T0_E )

######################################################################################
############################      SOLUTION TO PART B      ############################
############################ Classify based on Wx and Wy  ############################
######################################################################################

### Get Wx for the loop along the horizontal line indexed by rx: ###
def getWx(spins,rx):
    result=1
    indices = 2*L*rx + 2*np.arange(L)
    for i in indices:
        result = result*spins[i]
    return result

### Get Wy for the loop along the vertical line indexed by ry: ###
def getWy(spins,ry):
    result=1
    indices = 2*ry + 2*L*np.arange(L) + 1
    for i in indices:
        result = result*spins[i]
    return result

### Get Wx averaged over all horizontal lines of the lattice: ###
def getAveWx(spins):
    ave_Wx=0
    for i in range(L):
        ave_Wx = ave_Wx + getWx(spins,i)
    ave_Wx = ave_Wx/(1.0*L)
    return ave_Wx

### Get Wy averaged over all vertical lines of the lattice: ###
def getAveWy(spins):
    ave_Wy=0
    for i in range(L):
        ave_Wy = ave_Wy + getWy(spins,i)
    ave_Wy = ave_Wy/(1.0*L)
    return ave_Wy

num_T0_W = 0
### Loop over all configurations: ###
for c in range(N_configs):
    if c%1000==0:
        print('%d configs. checked' %c)
    ave_Wx = getAveWx(configs[c])
    ave_Wy = getAveWy(configs[c])
    if abs(ave_Wx) == 1 and abs(ave_Wy)==1:
        num_T0_W = num_T0_W + 1

print( 'Number topologically ordered (from Wx and Wy) = %d\n' %num_T0_W )
