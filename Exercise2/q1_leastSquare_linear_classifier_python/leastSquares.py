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

    # Add constant term
    num_samples = data.shape[0]
    x = np.concatenate([np.ones([num_samples, 1]), data], axis=1)

    # Compute pseudoinverse
    weight = np.linalg.lstsq(x, label, rcond=None)[0]
    weight, bias = weight[1:], weight[0]
    return weight, bias
