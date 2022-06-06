import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time as time

def initrandomtt(dataset,J, min, max,r=2):
    start = [np.random.randint(min,max,size=(r,1,J))]
    for i in range(len(dataset.columns.values)-3):
        start.append(np.array([np.random.randint(min,max,size=(r,r,J))] ))
    start.append(np.array([np.random.randint(min,max,size=(1,r,J))]))

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
    # print(f'{d, a}')
    for i in a:
        print(linalg.khatri_rao(X[i-1].transpose(), Gright) == linalg.khatri_rao(Gright,X[i-1].transpose()))
        Gright = linalg.khatri_rao(X[i-1].transpose(), Gright)
        Gright = np.reshape(tt[i-1], (R, R * I)).dot(Gright)


    Gright = linalg.khatri_rao(X[d].transpose(), Gright)

    return Gright

def leftSuperCore(tt,X,d):
    R = 2
    I = X[d].shape[1]
    D = len(tt)-1
    L = 1
    N = X[d].shape[0]


    Gleft = np.reshape(tt[0],(R,I)).dot(X[0].transpose())
    if d == 1:
        return Gleft

    for i in range(1,d):
        if (i % 2) == 1:
            Gleft = linalg.khatri_rao(X[i].transpose(), Gleft)
            Gleft = np.reshape(Gleft, (N, I*R)).dot(np.reshape(tt[i],(I*R, R)))
        if (i % 2) == 0:
            Gleft = linalg.khatri_rao(X[i].transpose(), Gleft.transpose()).transpose()
            Gleft = Gleft.dot(np.reshape(tt[i], (I * R, R)))

    if d ==D:
        Gleft = linalg.khatri_rao(X[D].transpose(),Gleft.transpose())

    return Gleft  # N x LRd

def getUL(tt, X, d):
    R = 2
    I = X[d].shape[1]
    D = len(tt)-1
    N = X[d].shape[0]

    if d == 0:
        superCore = rightSuperCore(tt, X, d) # R2 x N

        return np.reshape(superCore, (N, I*R)) # NL x LR2

    elif d==D:
        Gleft = leftSuperCore(tt, X, d)  # N x JLRd

        return np.reshape(Gleft, (N, I * R))


    Gright = rightSuperCore(tt, X, d)  # Rd1 x N
    Gleft = leftSuperCore(tt, X, d)  # N x JLRd

    if (d % 2) == 1:
        superCore = linalg.khatri_rao(Gright, Gleft)

    elif  (d % 2) == 0:
        superCore = linalg.khatri_rao(Gright, Gleft.transpose())

    return np.reshape(superCore, (N, R * I * R))  # NL x R2JR3


def updateCore(tt,X,d,y):

    U = getUL(tt,X,d)
    UTy = U.transpose().dot(np.reshape(y, (max(y.shape),1)))

    UTU = U.transpose().dot(U)

    w = pinv(UTU).dot(UTy)

    # print(f'd = {d}: U.shape = {U.shape}, y.shape = {y.shape}, UTy.shape = {UTy.shape}, UTU.shape = {UTU.shape}, w.shape = {w.shape}, UTU[0][0] = {U[0][0]} ')
    return w

def tt_ALS(tt,X,y,iter):
    D= len(tt)-1
    mpt = tt.copy()
    xss = ([[i for i in range(D+1)], [i for i in range(1,D)][::-1]])
    swipe = [x for xs in xss for x in xs]
    dims = []
    for i in range(len(tt)):
        dims.append(tt[i].shape)

    for i in range(0,iter):
        for j in range(len(swipe)):
            d = swipe[j]

            newCore = updateCore(tt,X,d,y)

            tt[d] = np.reshape(newCore,dims[d])

    return tt

def featurespace(dataset, p):
    cnames = dataset.columns.values
    res = [[[0 for _ in range(len(cnames))] for _ in range(len(dataset))] for _ in range(p)]

    for i in range(len(cnames)-1):
        flower = list(dataset.iloc[:, i])
        for j in range(len(dataset)):
            res[i][j] = np.array([flower[j]**i for i in range(p)])

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
    for i in [2,1]:
        Gright = linalg.khatri_rao(X[i].transpose(),Gright)             # Xi       # Gright = np.kron(np.reshape(Gright.transpose(),(1,300)),X) # Tweede
        Gright = np.reshape(tt[i], (R, R * I)).dot(Gright)
    superCore = linalg.khatri_rao(X[0].transpose(),Gright)
    superCore = np.reshape(tt[0],(1,R*I)).dot(superCore).transpose()

    return superCore


def t_test(dset, I, iter):
    train, test = train_test_split(dset,test_size=0.33)
    Xtrain = featurespace(train,I)
    Xtest = featurespace(test,I)
    y = yspace(train)
    yy = yclassifier(train, 0)
    # print(yy)
    # print(y)
    tt = initrandomtt(dset, I,0,3,r=2)
    traintt = tt_ALS(tt,Xtrain,yy,iter)

    model = supercore(traintt, Xtrain)
    # print(model)
    #compare
    count = 0
    for i in range(len(model)):
        if np.round(model[i],1) == yy[i]:
            count += 1
            # print(np.round(model[i],1), model[i], y[i])
    acc = (count/len(model)) * 100
    print(f'accuracy is: {acc}')

def pinv(M):
    return np.linalg.inv(M.T*M) * M.T


def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
                                          run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

# .... Define your model here ....


iris = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv')
irisdataset = iris.iloc[:,[1,2,3,4,5]]
# wine = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/winequality-white.csv',delimeter=';')
# print(wine)

#variables:
I = 4 #nauwkeurigheid
feature = 0
iter = 1
dataset = irisdataset

t_test(irisdataset, I, iter)


print(time.process_time())
# model = Sequential(t_test(irisdataset, I, iter))
# model.add(Dense(8, activation='softmax'))
# print(get_flops(model))