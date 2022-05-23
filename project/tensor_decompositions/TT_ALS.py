import numpy as np
import pandas as pd

data = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv')
s = data.size
def initrandomtt(size, rank = 2):
    over = size
    res = []
    while over != 0:
        res.append(np.random.rand(rank,rank))
        over -= rank*rank
    return res




print(initrandomtt(s))

