from PIL import Image

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


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
                print(p)
                y = np.transpose(x, p).copy()

                # decompose the full tensor
                for i in range(order -1):
                    # reshape residual tensor
                    m = ranks[i] * row_dims[i] * col_dims[i]
                    n = np.prod(row_dims[i + 1:]) * np.prod(col_dims[i + 1:])
                    y = np.reshape(y, [m, n])

                    # apply SVD in order to isolate modes
                    [u, s, v] = linalg.svd(y, full_matrices=False)

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

                    # define new TT core
                    ranks[i + 1] = u.shape[1]
                    cores.append(np.reshape(u, [ranks[i], row_dims[i], col_dims[i], ranks[i + 1]]))

                    # set new residual tensor
                    y = np.diag(s).dot(v)

                # define last TT core
                cores.append(np.reshape(y, [ranks[-2], row_dims[-1], col_dims[-1], 1]))

                for i in range(len(cores)):
                    print(f'norm of core {i+1} = {linalg.norm(cores[i])}')
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

        if self.ranks[0] != 1 or self.ranks[-1] != 1:
            raise ValueError("The first and last rank have to be 1!")

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
        full_tensor = full_tensor.reshape(p).transpose(q)

        return np.array(np.reshape(full_tensor, (512,512,3)))


def imshow(image):
    img = Image.fromarray(image)
    img.show()

img_dog = np.array(Image.open('dog.jpg'))
img_bab = np.array(Image.open('baboon.png'))
img1 = np.reshape(img_bab, (4,4,4,4,4,4,4,4,4,3))
# print(f'bab = {img_bab}')
# print(f'dog = {img_dog}')
dog = TT.full(TT(img1,threshold=0.2))
# print(img.info)
# print(dog.info)
# imshow(dog.astype(np.uint8))