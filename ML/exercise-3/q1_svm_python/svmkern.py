import numpy as np
from kern import kern
import cvxopt


def svmkern(X, t, C, p):
    norm = 5
    m,n = X.shape
    y = t.reshape(-1,1) * 1.
    X_dash = y * X
    H = kern(X_dash.T , X_dash.T, norm) * 1.0

    #Converting into cvxopt format - as previously
    P = cvxopt.matrix(H)
    q = cvxopt.matrix(-np.ones((m, 1)))
    G = cvxopt.matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.zeros(1))

    #Run solver
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    #==================Computing and printing parameters===============================#
    w = ((y * alphas).T @ X).reshape(-1,1)
    sv = (alphas > 1e-5).flatten()
    b = (y[sv] - np.dot(X[sv], w))[0]

    print(b, w)
    result, slack = [], []
    for x in X:
        calc = np.dot(w.T,x) + b
        if calc >= 0: 
            result.append(int(1))
        else:
            result.append(int(-1))
        slack.append(calc > C)
    
    result = np.array(result)
    slack = np.array(slack)









    # Non-Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                        (num_samples x dim)
    # t        : labeling                           (num_samples x 1)
    # C        : penalty factor the slack variables (scalar)
    # p        : order of the polynom               (scalar)
    #
    # OUTPUT:
    # sv       : support vectors (boolean)          (1 x num_samples)
    # b        : bias of the classifier             (scalar)
    # slack    : points inside the margin (boolean) (1 x num_samples)

    #####Insert your code here for subtask 2d#####
    return alphas, sv, b, result, slack
