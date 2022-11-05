import numpy as np


def knn(samples, k):
    
    pos = np.arange(-5, 5.0, 0.1)
    estDensity = []
    for i in range(len(pos)):
        x = pos[i]
        
        distances = np.sort([np.linalg.norm(x- xi) for xi in samples])
        
        k_neighbours = distances[k-1]
        v_star = 2 * k_neighbours
        N = len(pos)
        estDensity.append([x, k/(N*v_star)])
    
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    return np.array(estDensity)
