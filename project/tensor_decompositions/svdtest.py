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
    data = np.reshape(data,(4,6,5,5))
    tt = tensor_train(data, 4, verbose=False)

    return tt

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
for R in range(1,10):
    for I in range(1,10):
        try:
            shaper(vlak,R,I)
        except:
            print(R,I,'werkt niet')
        else:
            print(R,I)

