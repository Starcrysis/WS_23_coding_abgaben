import numpy as np


def kde(samples, h):
    pos = np.arange(-5, 5.0, 0.1)
    estDensity = []
    
    if(len(samples.shape) == 1):
        D = 1
    else:
        D = samples.shape[1]
        
    # def k(u): 
    #     return (1/(np.sqrt(2*np.pi*h**2)))*np.exp(-(u**2)/(2*h**2))
    
    erg_arr = []
    N = len(pos)
    
    for i in range(len(pos)):
        
        inner_sum = np.sum([1 / ( ( (2 * np.pi)**(D/2)) * h ) * np.exp(- ( abs(pos[i] - xi))**2 / 2*(h**2) )  for xi in samples])
        #K = np.sum([k(samples[i] - elem_2) for elem_2 in samples])
        p_x = 1/N * inner_sum
        estDensity.append([pos[i],p_x ])
    
            
        
    
     
    
    
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    return np.array(estDensity)
