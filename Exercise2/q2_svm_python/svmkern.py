import numpy as np
from kern import kern
import cvxopt


def svmkern(X, t, C, p):
    # Non-Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                        (num_samples x dim)
    # t        : labeling                           (num_samples x 1)
    # C        : penalty factor the slack variables (scalar)
    # p        : order of the polynom               (scalar)
    #
    # OUTPUT:
    # sv       : support vectors (boolean)          (1 x num_samples)
    # b        : bias of the classifier             (scalar)
    # slack    : points inside the margin (boolean) (1 x num_samples)

    N = X.shape[0]
    A = t.reshape([1, N])
    b = 0.0

    # Lower and upper bounds
    LB = np.zeros(N)
    UB = C * np.ones(N)

    H = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            H[i, j] = t[i] * t[j] * kern(X[i], X[j], p)

    f = -1 * np.ones(N)
    n = H.shape[1]
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([LB, UB])
    sol = cvxopt.solvers.qp(
        P=cvxopt.matrix(H),
        q=cvxopt.matrix(f),
        G=cvxopt.matrix(G),
        h=cvxopt.matrix(h),
        A=cvxopt.matrix(A),
        b=cvxopt.matrix(b),
    )

    alpha = np.array(sol["x"]).reshape((-1,))
    sv = np.where(alpha > 1e-6, True, False)
    slack = np.where(alpha > C - 1e-6, True, False)
    w = (alpha[sv] * t[sv]).dot(X[sv])
    b = np.mean(t[sv] - w.dot(X[sv].T))
    result = X.dot(w) + b

    return alpha, sv, b, result, slack
