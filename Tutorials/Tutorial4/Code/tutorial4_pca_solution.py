import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

#Specify font sizes for plots:
plt.rcParams['axes.labelsize']  = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['font.size']       = 18

modelName = "gaugeTheory" #can be "Ising" or "gaugeTheory"

#Parameters:
num_components = 2

### Loop over all lattice sizes: ###
for L in [20,40,80]:
    print("L=%d"%L)

    ### Read in the data from the files: ###
    X      = np.loadtxt("Data_Tutorial4/spinConfigs_%s_L%d.txt" %(modelName,L), dtype='uint8')
    
    if modelName == "Ising": # For the Ising model, the "labels" for visualization are the temperatures
        labels = np.loadtxt("Data_Tutorial4/temperatures_%s_L%d.txt" %(modelName,L), dtype='float')
    else: # For the gauge theory, the "labels" for visualization are 0 (T=0 phase) or 1 (T=infinity phase)
        labels = np.loadtxt("Data_Tutorial4/labels_%s_L%d.txt" %(modelName,L), dtype='float')

    ### Perform the PCA: ###
    (N_configs, N_spins) = X.shape
    X_cent = X - np.tile(np.mean(X, 0), (N_configs, 1))
    (lamb, P) = np.linalg.eig(np.dot(X_cent.T, X_cent))
    Y = np.dot(X_cent, P[:,0:num_components])
    
    ### PLOT FIGURE FOR PART C: ###
    plt.figure(1)
    ratios = lamb/np.sum(lamb)
    ratios = np.sort(ratios)[::-1] #The [::-1] is to get the ratios in reverse order (largest first)
    plt.semilogy(np.arange(N_spins), ratios, 'o-', label="L=%d"%L)

    ### PLOT FIGURE FOR PARTS A and B: ###
    plt.figure()
    sc = plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=40, cmap=plt.cm.coolwarm) #PART B
    #plt.scatter(Y[:, 0], Y[:, 1], s=40) #PART A
    plt.title("L=%d"%L)
    cb = plt.colorbar(sc, cmap=plt.cm.coolwarm)
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.savefig("y1y2_%s_L%d.pdf" %(modelName,L))

    ### PLOT FIGURE FOR PART D: ###
    plt.figure()
    plt.axes([0.17, 0.13, 0.81, 0.78]) #specify axes (for figure margins) in the format [xmin, ymin, xwidth, ywidth]
    plt.plot(np.arange(N_spins),np.abs(P[:,0]))
    plt.title("L=%d"%L)
    plt.xlabel("Component index")
    plt.ylabel("Absolute value of components of p1")
    plt.savefig("p1_%s_L%d.pdf"  %(modelName,L))

plt.figure(1)
plt.xlim([0,10])
plt.ylim([10**(-3),1])
plt.xlabel("Component index")
plt.ylabel("Explained variance ratio")
plt.legend()
plt.savefig("ratios_%s.pdf" %modelName)

#plt.show()
