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


def tensortrain(img, epsilon=0.1):
    img = np.reshape(np.asarray(img), (5,5,2,2,3,2))
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
    # g.append(len(g))
    # for i in range(len(g)):
    #     print(f'norm of core {i+1} = {linalg.norm(g[i])}')
    # print(f'norm of core = {linalg.norm(img)}')
    count = 0
    for core in g:
        for x in core:
            for y in x:
                for i in y:
                    count += 1
    print(f'{count} items in tt')
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



iris = '/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv'
indiaan = "/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/pima-indians-diabetes.csv"
wine = "/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/winequality-white.csv"


dataset = iris #iris #indiaan #wine

def ttest(data):
    data.drop(columns=data.columns[-1],
                    axis=1,
                    inplace=True)

    core, d, r, n = tensortrain(data)
    return core







