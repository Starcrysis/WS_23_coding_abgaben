import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
# import os
# os.environ["PATH"] += os.pathsep + ...
import cvxopt


def svmlin(X, t, C):
    # n_samples, n_features = X.shape
    # H = np.zeros([n_samples, n_samples])
    # for i in range(n_samples):
    #     for j in range(n_samples):
    #         H[i,j] = np.dot(X[i].astype(float).T, X[j].astype(float))
    
    # P = cvxopt.matrix(np.outer(t.astype(float),t.astype(float))*H)
    # q = cvxopt.matrix(-1.0 * np.ones(n_samples))
    # A = cvxopt.matrix(t.astype(float).T, (1,40), 'd')
    # b = cvxopt.matrix(0.0)
    # G = cvxopt.matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
    # h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples)*float(C))))
    
    # solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # # Lagrange multipliers
    # alphas = np.array(solution['x'])

    # #==================Computing and printing parameters===============================#
    # w = ((t * alphas).T @ X).reshape(-1,1).reshape(-1,1)
    # S = (alphas > 1e-10).flatten()
    # b = t[S] - np.dot(X[S], w)

    m,n = X.shape
    y = t.reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.0

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


    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (num_samples x dim)
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)


    #####Insert your code here for subtask 2a#####
    return alphas, sv, w, b, result, slack
