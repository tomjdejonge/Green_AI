from PIL import Image
import tensorly as tl
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from torchstat import stat
import torchvision.models as models
import tensorflow as tf


# Original file: T_train.py (method 1) ---------------------------------------------------------------------------------


class TT(object):

    def __init__(self, x, threshold=0, max_rank=np.infty, string=None):

        # initialize from list of cores
        if isinstance(x, list):

            # check if orders of list elements are correct
            if np.all([x[i].ndim == 4 for i in range(len(x))]):

                # check if ranks are correct
                if np.all([x[i].shape[3] == x[i + 1].shape[0] for i in range(len(x) - 1)]):

                    # define order, row dimensions, column dimensions, ranks, and cores
                    self.order = len(x)
                    self.row_dims = [x[i].shape[1] for i in range(self.order)]
                    self.col_dims = [x[i].shape[2] for i in range(self.order)]
                    self.ranks = [x[i].shape[0] for i in range(self.order)] + [x[-1].shape[3]]
                    self.cores = x

                else:
                    raise ValueError('Shapes of list elements do not match.')

            else:
                raise ValueError('List elements must be 4-dimensional arrays.')

        # initialize from full array   
        elif isinstance(x, np.ndarray):

            # check if order of ndarray is a multiple of 2
            if np.mod(x.ndim, 2) == 0:

                # show progress
                if string is None:
                    string = 'HOSVD'

                # define order, row dimensions, column dimensions, ranks, and cores
                order = len(x.shape) // 2
                row_dims = x.shape[:order]
                col_dims = x.shape[order:]
                ranks = [1] * (order + 1)
                cores = []

                # permute dimensions, e.g., for order = 4: p = [0, 4, 1, 5, 2, 6, 3, 7]
                p = [order * j + i for i in range(order) for j in range(2)]
                # print(p)
                y = np.transpose(x, p).copy()

                # decompose the full tensor
                for i in range(order - 1):
                    # reshape residual tensor
                    m = ranks[i] * row_dims[i] * col_dims[i]
                    n = np.prod(row_dims[i + 1:]) * np.prod(col_dims[i + 1:])
                    y = np.reshape(y, [m, n])

                    # apply SVD in order to isolate modes
                    [u, s, v] = linalg.svd(y, full_matrices=False)
                    # print(y.shape)
                    # rank reduction
                    if threshold != 0:
                        indices = np.where(s / s[0] > threshold)[0]
                        u = u[:, indices]
                        s = s[indices]
                        v = v[indices, :]
                    if max_rank != np.infty:
                        u = u[:, :np.minimum(u.shape[1], max_rank)]
                        s = s[:np.minimum(s.shape[0], max_rank)]
                        v = v[:np.minimum(v.shape[0], max_rank), :]
                    print(u.shape[1], indices)
                    # define new TT core
                    ranks[i + 1] = u.shape[1]
                    cores.append(np.reshape(u, [ranks[i], row_dims[i], col_dims[i], ranks[i + 1]]))

                    # set new residual tensor
                    y = np.diag(s).dot(v)
                    # print(y.shape)

                # define last TT core
                cores.append(np.reshape(y, [ranks[-2], row_dims[-1], col_dims[-1], 1]))
                # print(cores)

                # print(np.reshape(y, [ranks[-2], row_dims[-1], col_dims[-1], 1]))
                # print(ranks[-2],'a', row_dims[-1],'a', col_dims[-1], 1)
                # for i in range(len(cores)):
                #     print(f'norm of core {i+1} = {linalg.norm(cores[i])}')
                # print(f'norm of core = {linalg.norm(x)}')
                # initialize tensor train
                self.__init__(cores)


            else:
                raise ValueError('Number of dimensions must be a multiple of 2.')

        else:
            raise TypeError('Parameter must be either a list of cores or an ndarray.')

    def __repr__(self):
        return ('\n'
                'Tensor train with order    = {d}, \n'
                '                  row_dims = {m}, \n'
                '                  col_dims = {n}, \n'
                '                  ranks    = {r}'.format(d=self.order, m=self.row_dims, n=self.col_dims, r=self.ranks))

    def full(self):
        # print(self.ranks)
        # print(self.col_dims)
        # print(self.row_dims)
        if self.ranks[0] != 1 or self.ranks[-1] != 1:
            raise ValueError("The first and last rank have to be 1!")
        # print('cores = ',self.cores)
        # reshape first core
        full_tensor = self.cores[0].reshape(self.row_dims[0] * self.col_dims[0], self.ranks[1])

        for i in range(1, self.order):
            # contract full_tensor with next TT core and reshape
            full_tensor = full_tensor.dot(self.cores[i].reshape(self.ranks[i],
                                                                self.row_dims[i] * self.col_dims[i] * self.ranks[
                                                                    i + 1]))
            full_tensor = full_tensor.reshape(np.prod(self.row_dims[:i + 1]) * np.prod(self.col_dims[:i + 1]),
                                              self.ranks[i + 1])

        # reshape and transpose full_tensor
        p = [None] * 2 * self.order
        p[::2] = self.row_dims
        p[1::2] = self.col_dims
        q = [2 * i for i in range(self.order)] + [1 + 2 * i for i in range(self.order)]
        # print(f'q = {q}')
        # print(f'p = {p}')
        # print(full_tensor.shape)
        full_tensor = full_tensor.reshape(p).transpose(q)
        # print(full_tensor)
        return np.array(np.reshape(full_tensor, (512, 512, 3)))


def imshow(image):
    img = Image.fromarray(image)
    img.show()


# Original file: T_Traintex (method 2): --------------------------------------------------------------------------------


def tensortrain(img, epsilon=0.1):
    img = np.reshape(np.asarray(img), (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
    n = img.shape
    d = len(n)
    delta = (epsilon / np.sqrt(d - 1)) * np.linalg.norm(img)
    r = np.zeros(d + 1)
    r[0] = 1
    r[-1] = 1
    g = []
    c = img.copy()

    for k in range(d - 1):
        m = int(r[k] * n[k])  # r_(k-1)*n_k
        b = int(c.size / m)  # numel(C)/r_(k-1)*n_k
        c = np.reshape(c, [m, b])
        [U, S, V] = linalg.svd(c, full_matrices=False)
        V = V.transpose()
        S = np.diag(S)
        s = np.diagonal(S)
        s = np.reshape(s, (s.shape[0], 1))
        rank = 0
        error = np.linalg.norm(s[rank + 1])
        while error > delta:
            rank += 1
            error = np.linalg.norm(s[rank + 1:])
        r[k + 1] = rank + 1

        g.append(np.reshape(U[:, :int(r[k + 1])], [int(r[k]), int(n[k]), int(r[k + 1])]))
        p_1 = S[:int(r[k + 1]), :int(r[k + 1])]
        p_2 = V[:, :int(r[k + 1])]
        c = p_1 @ p_2.transpose()

    g.append(np.reshape(c, (int(r[- 2]), int(n[- 1]), int(r[-1]), 1)))
    g.append(len(g))
    # for i in range(len(g)):
    #     print(f'norm of core {i+1} = {linalg.norm(g[i])}')
    # print(f'norm of core = {linalg.norm(img)}')
    return g, d, r, n


def tt_reconstruction(cores, d, r, n):
    r = list(r.astype(np.uint))

    n = list(n)
    full_tensor = np.reshape(cores[0], (int(n[0]), int(r[1])))

    for k in range(1, d):
        full_tensor = full_tensor.dot(cores[k].reshape(int(r[k]), int(n[k]) * int(r[k + 1])))
        full_tensor = full_tensor.reshape(np.prod(n[:k + 1]), int(r[k + 1]))

    # q = [2* i for i in range(d//2)] + [1+2*i for i in range(d//2)]

    return np.array(np.reshape(full_tensor, (512, 512, 3)))


def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis('off')
    f.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.show()


def border(img, low, high, p=False):
    lowcount = 0
    highcount = 0

    for x in range(len(img)):
        for y in range(len(img[0])):
            for z in range(len(img[0][0])):
                if img[x][y][z] <= low:
                    lowcount += 1
                    img[x][y][z] = low - img[x][y][z]

                elif img[x][y][z] >= high:
                    highcount += 1
                    # print(B[x][y][z])
                    img[x][y][z] = 2 * high - img[x][y][z]
    # print(lowcount,highcount)
    return img


def check(tensor, tt, epsilon, d, r, n):
    error = np.linalg.norm(tt_reconstruction(tt, d, r, n) - tensor) / np.linalg.norm(tensor)
    n = len(tt) - 1
    normcheck = round((np.linalg.norm(tt[n]) - np.linalg.norm(tensor)) / np.linalg.norm(tensor))
    if normcheck == 0 and error < epsilon:
        print('goed gedaan tom en tex')
    else:
        print('slecht gedaan tom en tex')


# Original file: test.py (method 3): -----------------------------------------------------------------------------------


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

# ----------------------------------------------------------------------------------------------------------------------
# Fil in used method here:
method = 3
# ----------------------------------------------------------------------------------------------------------------------

def main():

    if method == 1:
# Using Original file T_train.py (method 1): ---------------------------------------------------------------------------

        img_bab = np.array(Image.open('images/baboon.png'))
        img_dog = np.array(Image.open('images/dog.jpg'))
        img1 = np.reshape(img_dog, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
        dog = TT.full(TT(img1, threshold=0.1))
        imshow(img_dog.astype(np.uint8))
        imshow(dog.astype(np.uint8))


    elif method == 2:
# Using Original file: T_Traintex (method 2): --------------------------------------------------------------------------

        img2 = Image.open('images/baboon.png')
        img = Image.open('images/dog.jpg')
        core, d, r, n = tensortrain(img)
        # print(core[-1])
        # for i in range(len(core)-1):
        #     print(i, 'aaa', core[i].shape)
        #     print(linalg.norm(core[i]))
        B = tt_reconstruction(core, d, r, n)
        B = border(np.array(B), 0, 255, p=True)
        new_image = Image.fromarray((B).astype(np.uint8), 'RGB')
        old_image = img
        check(img, core, 0.1, d, r, n)
        compare(old_image, new_image)


    elif method == 3:
# Using Original file: test.py (method 3): -----------------------------------------------------------------------------

        img2 = Image.open('images/baboon.png')
        img = Image.open('images/dog.jpg')
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

def flopcounter():
    model = models.resnet18()
    res = stat(model, (3, 224, 224))

    print(stat(model, (3, 224, 224)))

def calculate_flops():
    # Print to stdout an analysis of the number of floating point operations in the
    # model broken down by individual operations.
    tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')

if __name__ == "__main__":

    main()
    flopcounter()
    # calculate_flops()
