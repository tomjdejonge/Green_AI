import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.model_selection import train_test_split


def initrandomtt(dataset,J, r):
    start = [np.random.randint(0,2,size=(r,1,J))]
    for i in range(len(dataset.columns.values)-3):
        start.append(np.array([np.random.randint(0,2,size=(r,r,J))] ))
    start.append(np.array([np.random.randint(0,2,size=(1,r,J))]))

    return np.array(start,dtype=object)


def rightSuperCore(tt,X,d):
    R = 2
    I = X[d].shape[1]
    D = len(tt)-1
    L = 1
    N = X[d].shape[0]
    D = len(tt)-1
    a = np.arange(d+2, D+1, 1).tolist()[::-1]

    # print(f'tt[{D}] = {tt[D].shape}')
    Gright = np.dot(np.reshape(tt[D],(R,I)),X[D].transpose())            #X4                         # eerste
    print(f'a = {d, a}')
    for i in a:

        Gright = linalg.khatri_rao(Gright,X[i-1].transpose())         #Xi       # Gright = np.kron(np.reshape(Gright.transpose(),(1,300)),X) # Tweede
        Gright = np.reshape(tt[i-1],(R,R*I)).dot(Gright)

    if d ==0:

        Gright = linalg.khatri_rao(Gright, X[0].transpose())
        return Gright
    else:
        Gright = linalg.khatri_rao(Gright,X[d].transpose())  #X[d]

    return (Gright.transpose())

def leftSuperCore(tt,X,d):
    R = 2
    I = X[d].shape[1]
    D = len(tt)-1
    L = 1
    N = X[d].shape[0]
    # print(f'D = {D}')

    Gleft = np.reshape(tt[0],(R,I))
    if d == 1:
        # print(Gleft.shape, X.shape)
        # print(f'dot = {Gleft.dot(X.transpose()).shape}')
        Gleft = Gleft.dot(X[0].transpose())

        return Gleft.transpose()

    if d== 2:
        # print(f'Gleft,x = {Gleft.shape, X.shape}')
        Gleft = linalg.khatri_rao(Gleft,X[d])
        Gleft = np.reshape(Gleft,(N,I*R))
        Gleft = Gleft.dot(np.reshape(tt[d],(I*R,R)))

        return np.reshape(Gleft, (N, R))

    for i in range(1,d-1):
        # print(i)
        # print(Gleft.shape, X.shape)
        Gleft = linalg.khatri_rao(Gleft, X[d]).transpose()   # N x JLRi1
        Gleft = np.reshape(Gleft, (N*L, I*R))   # N x JLRi1 ->  NL x JRi1

        temp = np.reshape(tt[i+1],(I*R, R))  # Ri1 x J x Ri2 -> JRi1 x Ri2

        Gleft = Gleft.dot(temp)  # N x LRi2
        Gleft = np.reshape(Gleft, (N, L * R))  # NL x Ri2 -> N x LRi2

    if d == D:

        return linalg.khatri_rao(Gleft.transpose(), X[d].transpose()).transpose()  #X[d] # N x JLRd

    return Gleft  # N x LRd

def getUL(tt, X, d, L):
    R = 2
    I = X[d].shape[1]
    D = len(tt)-1
    L = 1
    D = len(tt)-1
    # print(L)
    N = X[d].shape[0]

    if d == 0:
        superCore = rightSuperCore(tt, X, d) # R2 x N

        return np.reshape(superCore, (N*L, I*R)) # NL x LR2

    elif d == 1:
        Gleft = leftSuperCore(tt, X, d)  # L x R2
        Gright = rightSuperCore(tt, X, d).transpose()  # J R3 x N


        superCore = linalg.khatri_rao(Gright, Gleft.transpose()) # JR3 x N kron R2 x L -> R2JR3 x LN

        return np.reshape(superCore, (N, R*I*R))  # NL x R2JR3

    elif d==D:
        Gleft = leftSuperCore(tt, X, d)  # N x JLRd

        return np.reshape(Gleft, (N, I*R))  # NL x J*Rd

    else:
        Gright = rightSuperCore(tt, X, d)  # Rd1 x N
        Gleft = leftSuperCore(tt, X, d)  # N x JLRd

    superCore = linalg.khatri_rao(Gright.transpose(),Gleft.transpose())

    return np.reshape(superCore, (N * L, R * I * R)) # N L x Rd J Rd1

def updateCore(tt,X,d,y):

    U = getUL(tt,X,d,1)

    y = np.reshape(y, (100,1))
    UTy = U.transpose().dot(y)

    UTU = U.transpose().dot(U)

    w = np.linalg.pinv(UTU).dot(UTy)

    print(f'd = {d}: U.shape = {U.shape}, y.shape = {y.shape}, UTy.shape = {UTy.shape}, UTU.shape = {UTU.shape}, w.shape = {w.shape}, UTU[0][0] = {U[0][0]} ')
    return w

def tt_ALS(tt,X,y,iter):
    D= len(tt)-1
    mpt = tt.copy()
    swipe = [0,1,2,3,2,1]
    dims = []
    for i in range(len(tt)):
        dims.append(tt[i].shape)

    for i in range(0,iter):
        for j in range(len(swipe)):
            d = swipe[j]

            newCore = updateCore(mpt,X,d,y)
            mpt[d] = np.reshape(newCore,dims[d])

    return mpt

def featurespace(dataset, p):
    cnames = dataset.columns.values
    res = [[[0 for _ in range(len(cnames))] for _ in range(len(dataset))] for _ in range(p)]

    for i in range(len(cnames)-1):
        for j in range(len(dataset)):

            flower = list(dataset.iloc[:, i])
            X = flower[j]
            Y = np.array([X**i for i in range(p)])
            res[i][j] = Y
    return np.asarray(res)

def hussel(dataset):
    train, test = train_test_split(dataset,test_size=0.33)
    return train,test

def setupy(dset):
    uv = list(dataset.iloc[:,-1].unique())
    res = np.zeros((len(dset), len(uv)))
    flower = list(dset.iloc[:, -1])

    for j in range(len(flower)):
        k = np.zeros(len(uv))
        k[uv.index(flower[j])] = 1
        res[j] = k

    return res

def yspace(dset):
    res = np.zeros((len(dset),1))
    uv = list(dataset.iloc[:,-1].unique())
    flower = list(dset.iloc[:, -1])
    for j in range(len(flower)):
        res[j] = uv.index(flower[j])
    return res

def supercore(tt, X):
    #contract
    D = len(tt)-1
    R = 2
    I = X[0].shape[1]

    Gright = np.dot(np.reshape(tt[D], (R, I)), X[D].transpose())
    for i in [3,2]:
        Gright = linalg.khatri_rao(Gright, X[i - 1].transpose())             # Xi       # Gright = np.kron(np.reshape(Gright.transpose(),(1,300)),X) # Tweede
        Gright = np.reshape(tt[i - 1], (R, R * I)).dot(Gright)
    superCore = linalg.khatri_rao(Gright, X[0].transpose())
    superCore = np.reshape(tt[0],(1,R*I)).dot(superCore)


    return superCore


def test(dset, I, iter):
    train, test = hussel(dset)
    Xtrain = featurespace(train,I)
    Xtest = featurespace(test,I)
    y = yspace(train)
    tt = initrandomtt(dataset, I, r=2)
    traintt = tt_ALS(tt,Xtrain,y,iter)

    model = supercore(traintt, Xtest)
    print(model)


iris = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv')
irisdataset = iris.iloc[:,[1,2,3,4,5]]
wine = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/winequality-white.csv')
# print(wine.iloc[:,[1,2,3,4,5]])

#variables:
I = 4 #nauwkeurigheid
feature = 0
iter = 3
dataset = irisdataset

test(dataset, I, iter)