import numpy as np


def getLogLikelihood(means, weights, covariances, X):
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

    if len(X.shape) > 1:
        N, D = X.shape
    else:
        N = 1
        D = X.shape[0]

    K = len(weights)

    logLikelihood = 0.0
    for i in range(N):
        p = 0
        for j in range(K):
            if N == 1:
                meansDiff = X - means[j]
            else:
                meansDiff = X[i, :] - means[j]
            cov = covariances[:, :, j]
            norm = 1.0 / ((2 * np.pi) ** (D / 2.0) * np.sqrt(np.linalg.det(cov)))

            p += (
                weights[j]
                * norm
                * np.exp(
                    -0.5 * ((meansDiff.T).dot(np.linalg.lstsq(cov.T, meansDiff.T)[0].T))
                )
            )

        logLikelihood += norm * np.log(p)

    return logLikelihood
