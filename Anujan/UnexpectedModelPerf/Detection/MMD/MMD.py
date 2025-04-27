import numpy as np
from sklearn.metrics import pairwise

def mmd_rbf(X, Y, gamma=0.5):
    """
    Calculates MMD using the RBF kernel.

    Args:
        X (numpy.ndarray): Data from the first distribution.
        Y (numpy.ndarray): Data from the second distribution.
        gamma (float): Kernel parameter.

    Returns:
        float: MMD value.
    """

    XX = pairwise.rbf_kernel(X, X, gamma)
    YY = pairwise.rbf_kernel(Y, Y, gamma)
    XY = pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# Example usage
# if __name__ == "__main__":
#     X = np.random.rand(100, 20)
#     Y = np.random.rand(100, 20) + 0.5  # Shifted distribution

#     print(X)
#     mmd_value = mmd_rbf(X, Y)
#     print(f"MMD value: {mmd_value}")