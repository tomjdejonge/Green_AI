from T_traintex import tensortrain, tt_reconstruction
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty

img = Image.open('dog.jpg')
img2 = Image.open('baboon.png')
core, d, r, n = tensortrain(img)
B = tt_reconstruction(core, d, r, n)
# print(B.shape)

A = np.array([[1,3,5],[2,4,6]])
Y = np.array([[[1,4,7,10],[2,5,8,11],[3,6,9,12]],
                [[13,16,19,22],[14,17,20,23],[15,18,21,24]]])
V = np.array([[1,2,3,4]])
print(V)
# Y = np.reshape(Y, (3,4,2))
# A = A.transpose(1,0)
# print(Y.shape)

def unfold(tensor, x):

    if x==1:
        tensor = tensor.transpose(1, 0, 2)
        shape = tensor.shape
        res = np.reshape(tensor, (shape[0],shape[1]*shape[2]))
    elif x==2:
        tensor = tensor.transpose(2,0,1)
        shape = tensor.shape
        res = np.reshape(tensor, (shape[0],shape[1]*shape[2]))
    elif x==3:
        tensor = tensor.transpose(0, 2, 1)
        shape = tensor.shape
        res = np.reshape(tensor, (shape[0], shape[1] * shape[2]))
    return res  # tl.reshape(tl.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))
# https://www.kolda.net/publication/TensorReview.pdf


def nmultiplication(tensor, matrix, n):
    tshape = tensor.shape
    mshape = matrix.shape
    # print(tshape,"aaa",mshape)
    res = matrix.dot(unfold(tensor,n))
    return np.reshape(res,(mshape[0],tshape[0],tshape[2]))


def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return ty.reshape(ty.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def fold(unfolded_tensor, mode, shape):

    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return ty.moveaxis(ty.reshape(unfolded_tensor, full_shape), 0, mode)

def mode_dot(tensor, matrix_or_vector, mode, transpose=False):

    # the mode along which to fold might decrease if we take product with a vector
    fold_mode = mode
    new_shape = list(tensor.shape)

    if len(matrix_or_vector.shape) == 2:  # Tensor times matrix
        # Test for the validity of the operation
        dim = 0 if transpose else 1
        if matrix_or_vector.shape[dim] != tensor.shape[mode]:
            raise ValueError(
                'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[dim]
                ))

        if transpose:
            matrix_or_vector = T.conj(T.transpose(matrix_or_vector))

        new_shape[mode] = matrix_or_vector.shape[0]
        vec = False

    elif T.ndim(matrix_or_vector) == 1:  # Tensor times vector
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
                         'Provided array of dimension {} not in [1, 2].'.format(T.ndim(matrix_or_vector)))

    res = np.dot(matrix_or_vector, unfold(tensor, mode))

    if vec:  # We contracted with a vector, leading to a vector
        return ty.reshape(res, shape=new_shape)
    else:  # tensor times vec: refold the unfolding
        return fold(res, fold_mode, new_shape)

print(mode_dot(Y, A, 1, transpose=False))