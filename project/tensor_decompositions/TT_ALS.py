import numpy as np
import pandas as pd

iris = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv')
dataset = iris.iloc[:,[1,2,3,4]]


def initrandomtt(I = 4, rank = 2):
    res = [[],[],[],[]]
    for i in range(1,len(res)-1):
        res[i] = np.array(np.random.rand(rank,rank,I))

    res[0] = np.array(np.random.rand(rank,1,I))
    res[-1] = np.array(np.random.rand(1,rank,I))

    return res


# for i in range(len(initrandomtt(iris.size))):
#     print(i,initrandomtt(iris.size)[i])
# print(initrandomtt(iris.size))

def rightsupercore(tt,X,d):
    tt = np.array(tt)
    X = np.array(X)
    D = len(tt)-1
    a = np.arange(d+2, D+1, 1).tolist()[::-1]
    # print(tt[D].shape, X.shape)
    Gright = tt[D]*X

    print('start',Gright.shape)
    for i in a:
        print(i, Gright.shape)
        b = np.reshape(np.matrix.flatten(Gright), (1,8))
        Gright = np.kron(b,X)
        Gright = np.reshape(Gright, (4,8))
        # print(Gright.shape)
        # print([(tt[i-1]).shape[0],np.prod((tt[i-1].shape[1:2]))])
        print((np.prod(tt[i-1].shape[1:3])))
        Gright = np.reshape(tt[i-1], ((tt[i-1]).shape[1],np.prod((tt[i-1].shape[1:3])))).dot(Gright.transpose())

         # Gright = reshape(tt.cores[i - 1], (size(tt.cores[i - 1])[1], prod(size(tt.cores[i - 1])[2:3]))) * Gright ' #Ri-1 x JRi * JRi x N -> Ri-1 x N
    if d ==0:
        return Gright
    Gright = np.kron(Gright,X)
    return Gright.tranpsose()

# X = [1,x,x**2,x**3]

def classifier(X):
    return [1, X, X**2, X**3]

X = classifier(0.1)
tt = initrandomtt(I = 4, rank = 2)

rightsupercore(tt, X ,0)


# Gright = reshape(tt.cores[i-1],(size(tt.cores[i-1])[1],prod(size(tt.cores[i-1])[2:3])))*Gright'