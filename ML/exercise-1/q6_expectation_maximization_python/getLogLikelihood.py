import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    
    logLikelihood = 0
    for n in range(len(X)):
        inner_sum = 0
        for k in range(len(weights)):
            D = len(covariances[:,:,k])
            inner_sum += weights[k] *  1 / (2*np.pi)**(D/2) * 1 / (np.linalg.det(covariances[:,:,k])**(1/2)) * np.exp(-1/2 * np.matmul(np.matmul(np.transpose(X[n]-means[k]), np.linalg.inv(covariances[:,:,k])), (X[n] - means[k])))
        logLikelihood += np.log(inner_sum)
    return logLikelihood
        
    
    
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    return logLikelihood

