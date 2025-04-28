import numpy as np
from sklearn.metrics import pairwise

def mmd_rbf(X, Y, gamma=0.5):
    """
    Given two distribtuions, calculates the MMD value between them.
    """

    XX = pairwise.rbf_kernel(X, X, gamma)
    YY = pairwise.rbf_kernel(Y, Y, gamma)
    XY = pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()
