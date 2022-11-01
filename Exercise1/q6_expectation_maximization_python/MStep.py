import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    N, D = X.shape
    K = gamma.shape[1]

    means = np.zeros((K, D))
    covariances = np.zeros((D, D, K))

    Nk = gamma.sum(axis=0)
    weights = Nk / N

    means = np.divide(gamma.T.dot(X), Nk[:, np.newaxis])

    for i in range(K):
        sigma = np.zeros((D, D))
        for j in range(N):
            meansDiff = X[j] - means[i]
            sigma = sigma + gamma[j, i] * np.outer(meansDiff.T, meansDiff)
        covariances[:, :, i] = sigma / Nk[i]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    return weights, means, covariances, logLikelihood
