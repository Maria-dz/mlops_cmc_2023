import numpy as np


def euclidean_distance(X, Y):
    a = np.repeat(np.sum(X ** 2, axis=1)[:, None], np.shape(Y)[0], axis=1)
    b = np.tile(np.sum(Y ** 2, axis=1), (np.shape(X)[0], 1))
    c = 2 * (X @ Y.T)
    return np.sqrt(a+b-c)


def cosine_distance(X, Y):
    a = X @ Y.T
    b = np.sum(X ** 2, axis=1)[:, None]
    c = np.sum(Y ** 2, axis=1)
    return 1 - a / np.sqrt(b * c)
