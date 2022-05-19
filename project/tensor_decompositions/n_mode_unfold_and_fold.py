import numpy as np
import tensorly as ty

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

X_kolda = np.array([[[1, 13],
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


def reorder(indices, mode):
    indices = list(indices)
    element = indices.pop(mode)
    return [element] + indices[::-1]


def unfold(tensor, mode=0):
    return np.transpose(tensor, reorder(range(tensor.ndim), mode)).reshape((tensor.shape[mode], -1))
    # return T.reshape(T.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def unfold_ty(tensor, mode=0):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def fold(unfolded, mode, shape):
    unfolded_indices = reorder(range(len(shape)), mode)
    original_shape = [shape[i] for i in unfolded_indices]
    unfolded = unfolded.reshape(original_shape)
    folded_indices = list(range(len(shape)-1, 0, -1))
    folded_indices.insert(mode, 0)
    return np.transpose(unfolded, folded_indices)


def fold_ty(unfolded, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded, full_shape), 0, mode)

# shape = X_kossaifi.shape
# unfolded = unfold_ty(X_kossaifi)
# refold = fold_ty(unfolded, 0, shape)
# print(refold)