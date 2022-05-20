import tensorly as tl
# from ._base_decomposition import DecompositionMixin
# from ..tt_tensor import validate_tt_rank, TTTensor
# from ..tt_matrix import validate_tt_matrix_rank, TTMatrix
# from ..utils import DefineDeprecated
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np


def validate_tt_rank(tensor_shape, rank='same', constant_rank=False, rounding='round',
                     allow_overparametrization=True):

    if rounding == 'ceil':
        rounding_fun = np.ceil
    elif rounding == 'floor':
        rounding_fun = np.floor
    elif rounding == 'round':
        rounding_fun = np.round
    else:
        raise ValueError(f'Rounding should be round, floor or ceil, but got {rounding}')

    if rank == 'same':
        rank = float(1)

    if isinstance(rank, float) and constant_rank:
        # Choose the *same* rank for each mode
        n_param_tensor = np.prod(tensor_shape) * rank
        order = len(tensor_shape)

        if order == 2:
            rank = (1, n_param_tensor / (tensor_shape[0] + tensor_shape[1]), 1)
            warnings.warn(
                f'Determining the tt-rank for the trivial case of a matrix (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor.')

        # R_k I_k R_{k+1} = R^2 I_k
        a = np.sum(tensor_shape[1:-1])

        # Border rank of 1, R_0 = R_N = 1
        # First and last factor of size I_0 R and I_N R
        b = np.sum(tensor_shape[0] + tensor_shape[-1])

        # We want the number of params of decomp (=sum of params of factors)
        # To be equal to c = \prod_k I_k
        c = -n_param_tensor
        delta = np.sqrt(b ** 2 - 4 * a * c)

        # We get the non-negative solution
        solution = int(rounding_fun((- b + delta) / (2 * a)))
        rank = rank = (1,) + (solution,) * (order - 1) + (1,)

    elif isinstance(rank, float):
        # Choose a rank proportional to the size of each mode
        # The method is similar to the above one for constant_rank == True
        order = len(tensor_shape)
        avg_dim = [(tensor_shape[i] + tensor_shape[i + 1]) / 2 for i in range(order - 1)]
        if len(avg_dim) > 1:
            a = sum(avg_dim[i - 1] * tensor_shape[i] * avg_dim[i] for i in range(1, order - 1))
        else:
            warnings.warn(
                f'Determining the tt-rank for the trivial case of a matrix (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor.')
            a = avg_dim[0] ** 2 * tensor_shape[0]
        b = tensor_shape[0] * avg_dim[0] + tensor_shape[-1] * avg_dim[-1]
        c = -np.prod(tensor_shape) * rank
        delta = np.sqrt(b ** 2 - 4 * a * c)

        # We get the non-negative solution
        fraction_param = (- b + delta) / (2 * a)
        rank = tuple([max(int(rounding_fun(d * fraction_param)), 1) for d in avg_dim])
        rank = (1,) + rank + (1,)

    else:
        # Check user input for potential errors
        n_dim = len(tensor_shape)
        if isinstance(rank, int):
            rank = [1] + [rank] * (n_dim - 1) + [1]
        elif n_dim + 1 != len(rank):
            message = 'Provided incorrect number of ranks. Should verify len(rank) == tl.ndim(tensor)+1, but len(rank) = {} while tl.ndim(tensor) + 1  = {}'.format(
                len(rank), n_dim + 1)
            raise (ValueError(message))

        # Initialization
        if rank[0] != 1:
            message = 'Provided rank[0] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[0] to 1.'.format(
                rank[0])
            raise ValueError(message)
        if rank[-1] != 1:
            message = 'Provided rank[-1] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[-1] to 1.'.format(
                rank[0])
            raise ValueError(message)

    if allow_overparametrization:
        return list(rank)
    else:
        validated_rank = [1]
        for i, s in enumerate(tensor_shape[:-1]):
            n_row = int(rank[i] * s)
            n_column = np.prod(tensor_shape[(i + 1):])  # n_column of unfolding
            validated_rank.append(min(n_row, n_column, rank[i + 1]))
        validated_rank.append(1)

        return validated_rank


def tensor_train(input_tensor, rank, verbose=False):

    rank = validate_tt_rank(tl.shape(input_tensor), rank=rank)
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)
    # print(rank)
    unfolding = input_tensor
    factors = [None] * n_dim

    # Getting the TT factors up to n_dim - 1
    for k in range(n_dim - 1):

        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k] * tensor_size[k])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        (n_row, n_column) = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])
        U, S, V = tl.partial_svd(unfolding, current_rank)
        rank[k + 1] = current_rank

        # Get kth TT factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k + 1]))

        if (verbose is True):
            print("TT factor " + str(k) + " computed with shape " + str(factors[k].shape))

        # Get new unfolding matrix for the remaining factors
        unfolding = tl.reshape(S, (-1, 1)) * V

    # Getting the last factor
    (prev_rank, last_dim) = unfolding.shape
    factors[-1] = tl.reshape(unfolding, (prev_rank, last_dim, 1))

    if (verbose is True):
        print("TT factor " + str(n_dim - 1) + " computed with shape " + str(factors[n_dim - 1].shape))

    return factors


def tensor_train_matrix(tensor, rank):

    order = tl.ndim(tensor)
    n_input = order // 2  # (n_output = n_input)

    if tl.ndim(tensor) != n_input * 2:
        msg = 'The tensor should have as many dimensions for inputs and outputs, i.e. order should be even '
        msg += f'but got a tensor of order tl.ndim(tensor)={order} which is odd.'
        raise ValueError(msg)

    in_shape = tl.shape(tensor)[:n_input]
    out_shape = tl.shape(tensor)[n_input:]

    if n_input == 1:
        # A TTM with a single factor is just a matrix...
        return TTMatrix([tensor.reshape(1, in_shape[0], out_shape[0], 1)])

    new_idx = list([idx for tuple_ in zip(range(n_input), range(n_input, 2 * n_input)) for idx in tuple_])
    new_shape = list([a * b for (a, b) in zip(in_shape, out_shape)])
    tensor = tl.reshape(tl.transpose(tensor, new_idx), new_shape)

    factors = tensor_train(tensor, rank).factors
    for i in range(len(factors)):
        factors[i] = tl.reshape(factors[i], (factors[i].shape[0], in_shape[i], out_shape[i], -1))

    return TTMatrix(factors)


def tt_to_tensor(factors):

    if isinstance(factors, (float, int)): #0-order tensor
        return factors

    full_shape = [f.shape[1] for f in factors]
    full_tensor = tl.reshape(factors[0], (full_shape[0], -1))

    for factor in factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    return tl.reshape(full_tensor, full_shape)

def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image1)
    plt.axis('off')
    f.add_subplot(1,2,2)
    plt.imshow(image2)
    plt.axis('off')
    plt.show()

img2 = Image.open('baboon.png')
img = Image.open('dog.jpg')
img = np.asarray(img)
img = np.reshape(img, (4,4,4,4,4,4,4,4,4,3))
core = tensor_train(img.astype(float),4)

for i in range(len(core)):
    print(core[i].shape)
# B = tt_reconstruction(core, d,r,n)
# B = np.array(B)
B = tt_to_tensor(core)
B = np.reshape(B, (512,512,3))
new_image = Image.fromarray((B).astype(np.uint8),'RGB')
compare(new_image, new_image)
