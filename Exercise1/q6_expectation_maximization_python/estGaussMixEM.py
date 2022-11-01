import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians
    # logLikelihood  : log-likelihood of the data given the model

    N = data.shape[1]

    weights = np.ones(K) / K
    covariances = np.zeros((N, N, K))

    kmeans = KMeans(n_clusters=K, n_init=10).fit(data)
    cluster_labels = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create initial covariance matrices
    for j in range(K):
        data_cluster = data[cluster_labels == j]
        min_dist = np.inf
        for i in range(K):
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(N) * min_dist

    # EM
    for i in range(n_iters):

        logL, gamma = EStep(means, covariances, weights, data)
        weights, means, covariances, newlogL = MStep(gamma, data)

        for j in range(K):
            covariances[:, :, j] = regularize_cov(covariances[:, :, j], epsilon)

        if abs(logL - newlogL) < 1:
            break

    return [weights, means, covariances]
