import numpy as np


def tensor_to_vec(tensor):
    return np.reshape(tensor, (-1,))


def vec_to_tensor(vec, shape):
    return np.reshape(vec, shape)