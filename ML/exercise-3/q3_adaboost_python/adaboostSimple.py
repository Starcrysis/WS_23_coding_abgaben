import numpy as np
from numpy.random import choice

from simpleClassifier import simpleClassifier

def calc_err(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def calc_alpha(error):
    return 1/2*np.log((1 - error) / error)

def update_weights(w_i, alpha, y, y_pred):
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

def predict_model(j, theta, X, Y):
    y_pred = []
    for elem in X:
        if elem[j] > theta:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred

def adaboostSimple(X, Y, K, nSamples):
    j, theta = simpleClassifier()

    for m in range(0, K):
        if m == 0: 
            w_i = np.ones(nSamples)*1/K
        else:
            w_i = update_weights(w_i, alpha_m, y, y_pred)
        
    
        






    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar) 
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK 	: voting weights (K x 1) - for each round
    # para		: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta

    #####Insert your code here for subtask 1c#####
    return alphaK, para
