import numpy as np
import tensorly as ty


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


def unfold_ty(tensor, mode=0):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def fold_ty(unfolded, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded, full_shape), 0, mode)

# shape = X_kossaifi.shape
# unfolded = unfold_ty(X_kossaifi)
# refold = fold_ty(unfolded, 0, shape)
# print(refold)