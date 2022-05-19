import numpy as np

X = np.array([[[0, 1],
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

# print(X[..., 0])


def reorder(indices, mode):
    indices = list(indices)
    element = indices.pop(mode)
    return [element] + indices[::-1]


def unfold(tensor, mode=0):
    return np.transpose(tensor, reorder(range(tensor.ndim), mode)).reshape((tensor.shape[mode], -1))


# print(unfold(X, 1))
