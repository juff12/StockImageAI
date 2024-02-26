import pandas as pd
import numpy as np
from hmmlearn import hmm

def forward(V, A, B, init_dist):
    alpha = np.zeros((V.shape[0], A.shape[0]))
    alpha[0, :] = init_dist * B[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(A.shape[0]):
            # matrix computation steps
            alpha[t, j] = alpha[t - 1] @ A[:, j] * B[j, V[t]]
    return alpha

def backward(V, A, B):
    beta = np.zeros((V.shape[0], A.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((A.shape[0]))
    
    # loop in backawrd way fro T-1 to
    # actual loop T-2 to 0 (python indexing)
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(A.shape[0]):
            beta[t, j] = (beta[t + 1] * B[:, V[t + 1]]) @ A[j, :]
    return beta

def baum_welch(V, A, B, init_dist, n_iter=100):
    M = A.shape[0]
    T = len(V)
    for n in range(n_iter):
        # estimation
        alpha = forward(V, A, B, init_dist)
        beta = backward(V, A, B)

        xi = np.zeros((M, M, T -1))
        for t in range(T - 1):
            # joint prob of observed data up to time t @ transistion prob * close prob as t+1 #
            # joint prob of observed data from time t+1
            denominator = (alpha[t, :].T @ A * B[:, V[t + 1]].T) @ beta[t + 1, :]
            for i in range(M):
                numerator = alpha[t, i] * A[i, :] * B[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
                
        gamma = np.sum(xi, axis=1)
        ## maximization step
        A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        
        # add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        
        K = B.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            B[:, l] = np.sum(gamma[:, V == 1], axis=1)
        
        B = np.divide(B, denominator.reshape((-1, 1)))
    return A, B