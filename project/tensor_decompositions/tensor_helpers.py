import numpy as np


X_kossaifi = np.array([[[0, 1],
               [2, 3],
               [4, 5],
               [6, 7]],

              [[8, 9],
               [10, 11],
               [12, 13],
               [14, 15]],

              [[16, 17],
               [18, 19],
               [20, 21],
               [22, 23]]])


def tensor_to_vec(tensor):
    return np.reshape(tensor, (-1,))


def vec_to_tensor(vec, shape):
    return np.reshape(vec, shape)