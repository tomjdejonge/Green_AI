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


    Gright = np.dot(np.reshape(tt[D],(R,I)),X[D].transpose())

    if d ==0:
        Gright = linalg.khatri_rao(X[2].transpose(), Gright)
        Gright = np.reshape(tt[2], (R, R * I)).dot(Gright)
        Gright = linalg.khatri_rao(X[1].transpose(), Gright)
        Gright = np.reshape(tt[1], (R, R * I)).dot(Gright)
        Gright = linalg.khatri_rao(X[0].transpose(), Gright)
        return Gright

    if d ==1:
        Gright = linalg.khatri_rao(X[2].transpose(), Gright)
        Gright = np.reshape(tt[2], (R, R * I)).dot(Gright)
        Gright = linalg.khatri_rao(X[1].transpose(), Gright)
        return Gright

    if d ==2:
        Gright = linalg.khatri_rao(X[2].transpose(), Gright)

        return Gright

    # for i in a:
    #     print(i)
    #     Gright = linalg.khatri_rao(Gright,X[i-1].transpose())         #Xi       # Gright = np.kron(np.reshape(Gright.transpose(),(1,300)),X) # Tweede
    #     # print(np.reshape(tt[i - 1], (R, R * I)).dot(Gright) == Gright.transpose().dot(np.reshape(tt[i - 1], (R, R * I)).transpose()))
    #     Gright = np.reshape(tt[i-1],(R,R*I)).dot(Gright)

    return Gright

def leftSuperCore(tt,X,d):
    R = 2
    I = X[d].shape[1]
    D = len(tt)-1
    L = 1
    N = X[d].shape[0]
    # print(f'D = {D}')

    Gleft = np.reshape(tt[0],(R,I)).dot(X[0].transpose())
    if d == 1:

        return Gleft

    if d== 2:

        Gleft = linalg.khatri_rao(X[1].transpose(), Gleft)
        Gleft = np.reshape(Gleft,(N,I*R)).dot(np.reshape(tt[1],(I*R,R)))

        return np.reshape(Gleft, (N, R))

    if d ==3:
        Gleft = linalg.khatri_rao(X[1].transpose(), Gleft)
        Gleft = np.reshape(Gleft, (N, I*R)).dot(np.reshape(tt[1],(I*R, R)))

        Gleft = linalg.khatri_rao(X[2].transpose(),Gleft.transpose()).transpose()
        Gleft = Gleft.dot(np.reshape(tt[2],(I*R,R)))
        Gleft = linalg.khatri_rao(X[3].transpose(),Gleft.transpose())

    return Gleft  # N x LRd

def getUL(tt, X, d):
    R = 2
    I = X[d].shape[1]
    D = len(tt)-1


    # print(L)
    N = X[d].shape[0]

    if d == 0:
        superCore = rightSuperCore(tt, X, d) # R2 x N

        return np.reshape(superCore, (N, I*R)) # NL x LR2

    elif d == 1:
        Gleft = leftSuperCore(tt, X, d)
        Gright = rightSuperCore(tt, X, d)
        superCore = linalg.khatri_rao(Gright, Gleft)
        return np.reshape(superCore, (N, R*I*R))  # NL x R2JR3

    elif d==D:
        Gleft = leftSuperCore(tt, X, d)  # N x JLRd

        return np.reshape(Gleft, (N, I*R))  # NL x J*Rd

    else:

        Gright = rightSuperCore(tt, X, d)  # Rd1 x N
        Gleft = leftSuperCore(tt, X, d)  # N x JLRd

    superCore = linalg.khatri_rao(Gright,Gleft.transpose())

    return np.reshape(superCore, (N, R * I * R)) # N L x Rd J Rd1

def updateCore(tt,X,d,y):

    U = getUL(tt,X,d)

    y = np.reshape(y, (100,1))
    UTy = U.transpose().dot(y)

    UTU = U.transpose().dot(U)

    w = pinv(UTU).dot(UTy)
    # print(d, w)
    # w = UTU @ (UTy)

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
            # print(i,j,newCore)
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

def yspace(dset):
    res = np.zeros((len(dset),1))
    uv = list(dataset.iloc[:,-1].unique())
    flower = list(dset.iloc[:, -1])
    for j in range(len(flower)):
        res[j] = uv.index(flower[j])
    return res

def yclassifier(dset, x):
    res = np.zeros((len(dset),1))
    uv = list(dataset.iloc[:,-1].unique())
    flower = list(dset.iloc[:, -1])

    for j in range(len(flower)):
        if uv.index(flower[j]) == x:
            res[j] = 1
        else:
            res[j] = -1
    return res

def supercore(tt, X):
    #contract
    D = len(tt)-1
    R = 2
    I = X[0].shape[1]

    Gright = np.dot(np.reshape(tt[D], (R, I)),X[D].transpose())
    for i in [3,2]:
        Gright = linalg.khatri_rao(X[i - 1].transpose(),Gright)             # Xi       # Gright = np.kron(np.reshape(Gright.transpose(),(1,300)),X) # Tweede
        Gright = np.reshape(tt[i - 1], (R, R * I)).dot(Gright)
    superCore = linalg.khatri_rao(X[0].transpose(),Gright)
    superCore = np.reshape(tt[0],(1,R*I)).dot(superCore).transpose()

    return superCore


def test(dset, I, iter):
    train, test = train_test_split(dset,test_size=0.33)
    Xtrain = featurespace(train,I)
    Xtest = featurespace(test,I)
    y = yspace(train)
    yy = yclassifier(train, 0)
    # print(yy)
    tt = initrandomtt(dset, I, r=2)
    traintt = tt_ALS(tt,Xtrain,y,iter)

    model = supercore(traintt, Xtrain)
    # print(model)
    #compare
    count = 0
    for i in range(len(model)):
        if np.round(model[i],1) == y[i]:
            count += 1
            # print(np.round(model[i],1), model[i], y[i])
    acc = (count/len(model)) * 100
    print(f'accuracy is: {acc}')

def pinv(M):
    return M.T * np.linalg.inv(M*M.T)

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