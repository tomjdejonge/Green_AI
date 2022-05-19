import numpy as np
from tensor_helpers import vec_to_tensor, tensor_to_vec
from n_mode_unfold_and_fold import reorder, unfold, unfold_ty, fold, fold_ty
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

U = np.array([[1, 3, 5],
             [2, 4, 6]])


def n_mode_product(tensor, matrix_or_vector, mode, transpose=False):
    # the mode along which to fold might decrease if we take product with a vector
    fold_mode = mode
    new_shape = list(tensor.shape)

    if np.ndim(matrix_or_vector) == 2:  # Tensor times matrix
        # Test for the validity of the operation
        dim = 0 if transpose else 1
        if matrix_or_vector.shape[dim] != tensor.shape[mode]:
            raise ValueError(
                'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[dim]
                ))

        if transpose:
            matrix_or_vector = np.conj(np.transpose(matrix_or_vector))

        new_shape[mode] = matrix_or_vector.shape[0]
        vec = False

    elif np.ndim(matrix_or_vector) == 1:  # Tensor times vector
        if matrix_or_vector.shape[0] != tensor.shape[mode]:
            raise ValueError(
                'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                ))
        if len(new_shape) > 1:
            new_shape.pop(mode)
        else:
            # Ideally this should be (), i.e. order-0 tensors
            # MXNet currently doesn't support this though..
            new_shape = []
        vec = True

    else:
        raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                         'Provided array of dimension {} not in [1, 2].'.format(np.ndim(matrix_or_vector)))

    res = np.dot(matrix_or_vector, unfold(tensor, mode))

    if vec:  # We contracted with a vector, leading to a vector
        return vec_to_tensor(res, shape=new_shape)
    else:  # tensor times vec: refold the unfolding
        return (fold(res, fold_mode, new_shape))


print(n_mode_product(X_kolda, U, 0))
