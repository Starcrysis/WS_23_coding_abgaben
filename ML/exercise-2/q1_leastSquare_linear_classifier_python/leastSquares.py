import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    A = []
    for i in range(len(data)):
        arr = [1]
        for elem in data[i][:]:
            arr.append(elem)
        A.append(arr)
    
    A = np.array(A)
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),label)

    bias = alpha[0]
    weight = alpha[1:]

    return weight, bias
