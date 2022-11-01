import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    # bg - background, fg - foreground
    weight_bg, means_bg, cov_bg = estGaussMixEM(ndata, K, n_iter, epsilon)
    weight_fg, means_fg, cov_fg = estGaussMixEM(sdata, K, n_iter, epsilon)

    height, width, _ = img.shape

    bg = np.ndarray((height, width))
    fg = np.ndarray((height, width))

    for h in range(height):
        for w in range(width):
            bg[h, w] = np.exp(
                getLogLikelihood(means_bg, weight_bg, cov_bg, img[h, w, :])
            )
            fg[h, w] = np.exp(
                getLogLikelihood(means_fg, weight_fg, cov_fg, img[h, w, :])
            )

    result = fg / bg
    result = np.where(result > theta, 1, 0)

    return result
