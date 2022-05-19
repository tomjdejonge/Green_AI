import numpy as np
from n_mode_unfold import reorder, unfold

X = np.array([[[1, 13],
               [4, 16],
               [7, 19],
               [10, 22]],

              [[2, 14],
               [5, 17],
               [8, 20],
               [11, 23]],

              [[3, 15],
               [6, 18],
               [9, 21],
               [12, 24]]])

U = np.array([[1, 3, 5],
             [2, 4, 6]])

# print(f'Frontal slices of X:\n', X[..., 0], '\n',  X[..., 1])
# for i in range(3):
#     print(f'Mode {i+1} unfold of X:\n', unfold(X, i))


def n_mode_product(tensor, matrix, mode=0):
    res = []

    return matrix @ unfold(tensor, mode)


def nmultiplication(tensor, matrix, n):
    res = matrix.dot(unfold(tensor,n))
    return np.reshape(res,2,2,4)


# X = A.dot(unfold(Y,1))
# print(np.reshape(X,(2,2,4)))
# W = V.dot(unfold(Y,2))
# print(np.reshape(W,(3,2)))

Z = (n_mode_product(X, U, 0))
