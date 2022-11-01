import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix

    regularized_cov = np.matrix(covariance + epsilon * np.eye(*covariance.shape))
    regularized_cov = (regularized_cov + regularized_cov.H) / 2
    return regularized_cov
