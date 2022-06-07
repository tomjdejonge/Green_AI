from PIL import Image
import tensorly as tl
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import pandas as pd



def datareader(location):
    df_comma = pd.read_csv(location, nrows=1, sep=",")
    df_semi = pd.read_csv(location, nrows=1, sep=";")
    if df_comma.shape[1] > df_semi.shape[1]:
        sepp = ','
    else:
        sepp = ';'
    dframe = pd.read_csv(location, sep=sepp)

    if len(dframe.iloc[:,0]) == dframe.iloc[-1,0] or len(dframe.iloc[:,0]) == dframe.iloc[-1,0] - 1:
        dframe.drop(columns=dframe.columns[0],
                axis=1,
                inplace=True)
    return dframe


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

                    # define new TT core
                    ranks[i + 1] = u.shape[1]
                    cores.append(np.reshape(u, [ranks[i], row_dims[i], col_dims[i], ranks[i + 1]]))

                    # set new residual tensor
                    y = np.diag(s).dot(v)


                # define last TT core
                cores.append(np.reshape(y, [ranks[-2], row_dims[-1], col_dims[-1], 1]))

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

        return full_tensor

iris = '/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv'
indiaan = "/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/pima-indians-diabetes.csv"
wine = "/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/winequality-white.csv"


dataset = iris #iris #indiaan #wine
data = datareader(dataset)

def ttest(data):
    data.drop(columns=data.columns[-1],
                    axis=1,
                    inplace=True)
    data = data.values.tolist()
    data = np.reshape(data,(5,5,4,6))
    dog = TT.full(TT(data, threshold=0.1))

    return dog

def initrandomtt(dataset,J, min, max,r):
    start = [np.random.randint(min,max,size=(r,1,J))]
    for i in range(len(dataset.columns.values)-3):
        start.append(np.array([np.random.randint(min,max,size=(r,r,J))] ))
    start.append(np.array([np.random.randint(min,max,size=(1,r,J))]))

    return np.array(start,dtype=object)


def flatten(ttest):
    listt = []
    for i in ttestt:
        for j in i:
            for k in j:
                for x in k:
                    listt.append(x)
    return listt

def shaper(tt, R, I):
    # print(len(tt))
    temp1 = []
    for i in range(R*I):
        temp1.append(tt.pop(i))
    first = np.reshape(temp1, (R,1,I))
    temp2 = []
    for i in range(R * I):
        temp2.append(tt.pop(i))
    last = np.reshape(temp2, (1,R,I))
    over = len(tt)
    midden = []
    while len(tt) != 0:
        temp3 = []
        for i in range(R * I):
            temp3.append(tt.pop(i))
        midden.append(np.reshape(temp3, (R, R, I)))

ttestt = ttest(data)
vlak = (flatten(ttest))
for R in range(10):
    for I in range(10):
        try:
            shaper(vlak,R,I)
        except:
            print(R,I,'werkt niet')
        else:
            print(R,I)
        finally:
            print('niks')
