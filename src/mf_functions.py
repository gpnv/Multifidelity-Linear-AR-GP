import numpy as np

def forrester_high(X):
    ndim = X.shape[1]
    term1 = (6*X - 2)**2
    term2 = np.sin(12*X - 4)
    return np.sum(term1 * term2, axis=1) / ndim


def forrester_low(X, *, A=0.5, B=10, C=-5):
    ndim = X.shape[1]
    term1 = A*forrester_high(X)
    term2 = B*(X - 0.5)
    term3 = C

    return term1 + (np.sum(term2, axis=1) / ndim) + term3

