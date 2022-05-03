# -*- coding: utf-8 -*-


import time as _time
import numpy as np
from scipy import linalg
import matplotlib.image as image


class TT(object):
    """
    Tensor train class
    Tensor trains [1]_ are defined in terms of different attributes. That is, a tensor train with order ``d`` is
    given by a list of 4-dimensional tensors
        ``[cores[0] , ..., cores[d-1]]``,
    where ``cores[i]`` is an ndarry with dimensions
        ``ranks[i] x row_dims[i] x col_dims[i] x ranks[i+1]``.
    There is no distinguish between tensor trains and tensor trains operators, i.e. a classical tensor train is
    represented by cores with column dimensions equal to 1.
    An instance of the tensor train class can be initialized either from a list of cores, i.e. ``t = TT(cores)``
    where ``cores`` is a list as described above, or from a full tensor representation, i.e. ``t = TT(x)`` where
    ``x`` is an ndarray with dimensions
        ``row_dims[0] x ... x row_dims[-1] x col_dims[0] x ... x col_dims[-1]``.
    In the latter case, the tensor is decomposed into the TT format. For more information on the implemented tensor
    operations, we refer to [2]_.
    Attributes
    ----------
    order : int
        order of the tensor train
    row_dims : list[int]
        list of the row dimensions of the tensor train
    col_dims : list[int]
        list of the column dimensions of the tensor train
    ranks : list[int]
        list of the ranks of the tensor train
    cores : list[np.ndarray]
        list of the cores of the tensor train
    Methods
    -------
    print(t)
        string representation of tensor trains
    +
        sum of two tensor trains
    -
        difference of two tensor trains
    *
        multiplication of tensor trains and scalars
    @/dot(t,u)
        multiplication of two tensor trains
    tensordot
        index contraction between two tensortrains
    rank_tensordot
        index contraction between TT and matrix along the rank-dimension
    concatenate
        concatenate cores of two TT
    transpose(t)
        transpose of a tensor train
    rank_transpose
        rank-transpose of a tensor train
    conj
        complex conjugate of a tensor train
    isoperator(t)
        check is given tensor train is an operator
    copy(t)
        deep copy of a tensor train
    element(t, indices)
        element of t at given indices
    full(t)
        convert tensor train to full format
    matricize(t)
        matricization of a tensor train
    ortho_left(t)
        left-orthonormalization of a tensor train
    ortho_right(t)
        right-orthonormalization of a tensor train
    ortho(t)
        left- and right-orthonormalization of a tensor train
    norm(t)
        norm of a tensor train
    tt2qtt
        conversion from TT format into QTT format
    qtt2tt
        conversion from QTT format into TT format
    svd
        Computation of a global SVD of a tensor train
    pinv
        Computation of the pseudoinverse of a tensor train
    References
    ----------
    .. [1] I. V. Oseledets, "Tensor-Train Decomposition", SIAM Journal on Scientific Computing 33 (5), 2011
    .. [2] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
           Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017

    Examples
    --------
    Construct tensor train from list of cores:
    Construct tensor train from ndarray:
    >>> import numpy as np
    >>> from scikit_tt.tensor_train import TT

    """

    def __init__(self, x, threshold=0, max_rank=np.infty, progress=False, string=None):
        """
        Parameters
        ----------
        x : list[np.ndarray] or np.ndarray
            either a list[TT] cores or a full tensor
        threshold : float, optional
            threshold for reduced SVD decompositions, default is 0
        max_rank : int, optional
            maximum rank of the left-orthonormalized tensor train, default is np.infty
        Raises
        ------
        TypeError
            if x is neither a list of ndarray nor a single ndarray
        ValueError
            if list elements of x are not 4-dimensional tensors or shapes do not match
        ValueError
            if number of dimensions of the ndarray x is not a multiple of 2
        """

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

                    # rank reduction
                    if threshold != 0 or max_rank != np.infty:
                        self.ortho(threshold=threshold, max_rank=max_rank)

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
                start_time = utl.progress(string, 0, show=progress)

                # define order, row dimensions, column dimensions, ranks, and cores
                order = len(x.shape) // 2
                row_dims = x.shape[:order]
                col_dims = x.shape[order:]
                ranks = [1] * (order + 1)
                cores = []

                # permute dimensions, e.g., for order = 4: p = [0, 4, 1, 5, 2, 6, 3, 7]
                p = [order * j + i for i in range(order) for j in range(2)]
                y = np.transpose(x, p).copy()

                # decompose the full tensor
                for i in range(order - 1):
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

                    # show progress
                    utl.progress(string, 100 * (i + 1) / order, cpu_time=_time.time() - start_time, show=progress)

                # define last TT core
                cores.append(np.reshape(y, [ranks[-2], row_dims[-1], col_dims[-1], 1]))

                # initialize tensor train
                self.__init__(cores)

                # show progress
                utl.progress(string, 100, cpu_time=_time.time() - start_time, show=progress)

            else:
                raise ValueError('Number of dimensions must be a multiple of 2.')

        else:
            raise TypeError('Parameter must be either a list of cores or an ndarray.')

    def __repr__(self):
        """
        String representation of tensor trains
        Print the attributes of a given tensor train.
        """

        return ('\n'
                'Tensor train with order    = {d}, \n'
                '                  row_dims = {m}, \n'
                '                  col_dims = {n}, \n'
                '                  ranks    = {r}'.format(d=self.order, m=self.row_dims, n=self.col_dims, r=self.ranks))


img=image.imread('dog.jpg')
print(TT(img))