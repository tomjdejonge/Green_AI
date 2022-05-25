import numpy as np
import pandas as pd
from scipy import linalg

iris = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv')
dataset = iris.iloc[:,[1,2,3,4]]


def initrandomtt2(I = 4, rank = 2):
    res = [[],[],[],[]]
    for i in range(1,len(res)-1):
        res[i] = np.array(np.random.rand(rank,rank,I))

    res[0] = np.array(np.random.rand(rank,1,I))
    res[-1] = np.array(np.random.rand(1,rank,I))

    return res

def initrandomtt(I=4, r=2):
    res = np.array([np.random.rand(r,1,I,),np.random.rand(r,r,I),np.random.rand(r,r,I),np.random.rand(1,r,I)],dtype=object)

    return res

def rightsupercore(tt,X,d):

    D = len(tt)-1
    a = np.arange(d+2, D+1, 1).tolist()[::-1]
    Gright = np.dot(np.reshape(tt[D],(tt[D].shape[1:3])),X.transpose()) # eerste

    for i in a:
        Gright = linalg.khatri_rao(np.reshape(Gright,(2,150)),X.transpose())  # Gright = np.kron(np.reshape(Gright.transpose(),(1,300)),X) # Tweede
        Gright = np.reshape(tt[i-1], ((tt[i-1]).shape[1],np.prod((tt[i-1].shape[1:3])))).dot(Gright) # laatste

    if d ==0:
        return Gright
    else:
        Gright = linalg.khatri_rao(Gright,X.transpose())
    return Gright.transpose()

def leftsupercore(tt,X,d):
    D = tt.shape
    r2,r1,L = tt[0].shape
    # print(r2,r1,L)
    Gleft = np.reshape(tt[0],(L,r2))
    # print(Gleft.shape)
    N = X.size
    if d == 1:
        return Gleft
    r3,r2,J = tt[1].shape
    Gleft = Gleft.dot(np.reshape(tt[2],(r2,J*r3)))
    Gleft = np.reshape(Gleft, (J,L*r3))
    Gleft = X.dot(Gleft)
    # print(Gleft.shape)
    if d == 3:
        return Gleft
    for i in range(2,d-2):
        print(i)
        Ri1, Ri, J = tt.cores[i].shape

    return None


def classifier2(X):
    return np.array([1, X, X**2, X**3])

def classifier(dataset, feature):
    # dataset = list(dataset)
    res = np.zeros((150,4))
    flower = dataset.iloc[:,feature]
    for j in range(len(flower)):
        X = flower[j]
        Y = np.array([1, X, X**2, X**3])
        res[j] = Y

    return res

X = classifier(dataset,1)
tt = initrandomtt(I = 4, r = 2)

# print(rightsupercore(tt, X ,1).shape)
leftsupercore(tt,X,0)

# Gright = reshape(tt.cores[i-1],(size(tt.cores[i-1])[1],prod(size(tt.cores[i-1])[2:3])))*Gright'