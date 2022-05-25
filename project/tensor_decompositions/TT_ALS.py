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
    # print(D)
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
    if d == 2:
        return Gleft

    for i in range(1,d-1):
        # print(i)
        Ri1, Ri, J = tt[i].shape
        # print(Gleft.shape, X.shape, Ri1, Ri, J)
        Gleft = linalg.khatri_rao(np.transpose(Gleft), X.transpose()).transpose()   # N x JLRi1
        N, JLRi1 = Gleft.shape
        L = JLRi1 // (J*Ri)
        # print(f'L = {L}')
        Gleft = np.reshape(Gleft, (N*L, J*Ri))   # N x JLRi1 ->  NL x JRi1
        Ri2, Ri1, J = (tt[i+1]).shape
        print(Ri1, J, Ri2)
        temp = np.reshape(tt[i+1],(J*Ri1, Ri2))  # Ri1 x J x Ri2 -> JRi1 x Ri2
        print(Gleft.shape, temp.shape)
        Gleft = Gleft.dot(temp)  # N x LRi2
        Gleft = np.reshape(Gleft, (N, L * Ri2))  # NL x Ri2 -> N x LRi2

    if d == D:
        return linalg.khatri_rao(Gleft, X)  # N x JLRd

    return Gleft  # N x LRd

def classifier2(X):
    return np.array([1, X, X**2, X**3])

def featurespace(dataset, feature, p):
    # dataset = list(dataset)

    res = np.zeros((len(dataset),p))
    # print(res.shape)
    flower = dataset.iloc[:,feature]
    for j in range(len(flower)):
        X = flower[j]
        Y = np.array([1, X, X**2, X**3])

        res[j] = Y
    return res

#datasplit

X = featurespace(dataset,1,4)
tt = initrandomtt(I = 4, r = 2)

# print(rightsupercore(tt, X ,1).shape)
# print(0, leftsupercore(tt,X,0).shape)
# print(1, leftsupercore(tt,X,1).shape)
# print(2, leftsupercore(tt,X,2).shape)
print(3, leftsupercore(tt,X,3).shape)

# Gright = reshape(tt.cores[i-1],(size(tt.cores[i-1])[1],prod(size(tt.cores[i-1])[2:3])))*Gright'