import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    N, D = X.shape
    K = len(weights)

    gamma = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            means_diff = X[i] - means[j]
            cov = covariances[:, :, j]
            norm = 1.0 / ((2 * np.pi) ** (D / 2) * np.sqrt(np.linalg.det(cov)))
            gamma[i, j] = (
                weights[j]
                * norm
                * np.exp(
                    -0.5 * (means_diff.T.dot(np.linalg.lstsq(cov.T, means_diff.T)[0].T))
                )
            )
        gamma[i] /= gamma[i].sum()

    return [logLikelihood, gamma]
