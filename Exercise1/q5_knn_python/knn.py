import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    N = len(samples)
    pos = np.arange(-5, 5.0, 0.1)

    dists = np.sort(np.abs(pos[np.newaxis, :] - samples[:, np.newaxis]), axis=0)
    res = (k / (2 * N)) / dists[k - 1, :]

    estDensity = np.stack((pos, res), axis=1)
    return estDensity
