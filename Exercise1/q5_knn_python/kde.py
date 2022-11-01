from random import gauss
import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    N = len(samples)
    pos = np.arange(-5, 5.0, 0.1)

    norm = np.sqrt(2 * np.pi) * h * N
    gauss = np.exp(-((pos[np.newaxis, :] - samples[:, np.newaxis]) ** 2) / (2 * h**2))
    res = np.sum(gauss, axis=0) / norm

    estDensity = np.stack((pos, res), axis=1)
    return estDensity
