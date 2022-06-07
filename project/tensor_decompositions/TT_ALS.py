import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.model_selection import train_test_split
import time as time
from matplotlib import pyplot as plt
from svdtest import ttest


def initrandomtt(dataset,J, min, max,r):
    start = [np.random.randint(min,max,size=(r,1,J))]
    for i in range(len(dataset.columns.values)-3):
        start.append(np.array([np.random.randint(min,max,size=(r,r,J))] ))
    start.append(np.array([np.random.randint(min,max,size=(1,r,J))]))

    return np.array(start,dtype=object)


def rightSuperCore(tt,X,d,R):

    I = X[d].shape[1]
    D = len(tt)-1
    L = 1
    N = X[d].shape[0]
    D = len(tt)-1
    a = np.arange(d+2, D+1, 1).tolist()[::-1]

    Gright = np.dot(np.reshape(tt[D],(R,I)),X[D].transpose())
    # print(f'{d, a}')
    for i in a:
        Gright = linalg.khatri_rao(Gright, X[i-1].transpose())
        Gright = np.reshape(tt[i-1], (R, R*I)).dot(Gright)

    Gright = linalg.khatri_rao(Gright, X[d].transpose())

    return Gright
"""  if d ==0:
        Gright = linalg.khatri_rao(X[2].transpose(), Gright)
        Gright = np.reshape(tt[2], (R, R * I)).dot(Gright)
        Gright = linalg.khatri_rao(X[1].transpose(), Gright)
        Gright = np.reshape(tt[1], (R, R * I)).dot(Gright)
        Gright = linalg.khatri_rao(X[0].transpose(), Gright)
        return Gright"""

def leftSuperCore(tt,X,d, R):
    I = X[d].shape[1]
    D = len(tt)-1
    L = 1
    N = X[d].shape[0]

    # print(X[0].dot(np.reshape(tt[0],(I,R))) == np.reshape(tt[0],(R,I)).dot(X[0].transpose()))
    Gleft = np.reshape(tt[0],(R,I)).dot(X[0].transpose())
    if d == 1:
        return Gleft
    for i in range(1,d):

        Gleft = linalg.khatri_rao(Gleft, X[i].transpose())
        Gleft = np.reshape(Gleft, (N, I*R)).dot(np.reshape(tt[i],(I*R, R)))
        Gleft = np.reshape(Gleft, (min(Gleft.shape),max(Gleft.shape)))

    if d ==D:
        Gleft = linalg.khatri_rao(Gleft,X[D].transpose())

    return Gleft  # N x LRd

def getUL(tt, X, d,R):

    I = X[d].shape[1]
    D = len(tt)-1
    N = X[d].shape[0]

    if d == 0:
        superCore = rightSuperCore(tt, X, d,R) # R2 x N

        return np.reshape(superCore, (N, I*R)) # NL x LR2

    elif d==D:
        Gleft = leftSuperCore(tt, X, d,R)  # N x JLRd

        return np.reshape(Gleft, (N, I * R))


    Gright = rightSuperCore(tt, X, d,R)  # Rd1 x N
    Gleft = leftSuperCore(tt, X, d,R)  # N x JLRd

    Gleft = np.reshape(Gleft, (min(Gleft.shape), max(Gleft.shape)))
    Gright = np.reshape(Gright, (min(Gright.shape), max(Gright.shape)))
    superCore = linalg.khatri_rao(Gright, Gleft)

    return np.reshape(superCore, (N, R * I * R))  # NL x R2JR3


def updateCore(tt,X,d,y,R):

    U = getUL(tt,X,d,R)
    # print(d,U)
    UTy = U.transpose().dot(np.reshape(y, (max(y.shape),1)))

    UTU = U.transpose().dot(U)

    w = linalg.inv(UTU).dot(UTy)

    #np.linalg.inv(M.T*M) * M.T
    # print(f'd = {d}: U.shape = {U.shape}, y.shape = {y.shape}, UTy.shape = {UTy.shape}, UTU.shape = {UTU.shape}, w.shape = {w.shape}, UTU[0][0] = {U[0][0]} ')
    return w

def tt_ALS(tt,X,y,R):
    D= len(tt)-1
    xss = ([[i for i in range(D+1)], [i for i in range(1,D)][::-1]])           # ([[i for i in range(D+1)], [i for i in range(1,D)][::-1]])  #([[i for i in range(1,D)][::-1],[i for i in range(D+1)]])
    swipe = [x for xs in xss for x in xs]
    dims = []

     # [collect(1:D)..., collect(D-1:-1:2)...] = [0,1,2,3,2,1]??

    for i in range(len(tt)):
        dims.append(tt[i].shape)

    for j in range(len(swipe)):
        d = swipe[j]

        newCore = updateCore(tt,X,d,y,R)
        tt[d] = np.reshape(newCore,dims[d])
        # print(tt)

    return tt

def featurespace(dtset, p):
    cnames = dtset.columns.values
    res = [[[0 for _ in range(p)] for _ in range(len(dtset))] for _ in range(len(cnames))]
    for i in range(len(cnames)-1):
        flower = list(dtset.iloc[:, i])
        for j in range(len(dtset)):

            res[i][j] = np.array([flower[j]**i for i in range(p)])

    return np.asarray(res)

def yspace(dset):
    res = np.zeros((len(dset),1))
    uv = list(dset.iloc[:,-1].unique())
    flower = list(dset.iloc[:, -1])
    for j in range(len(flower)):
        res[j] = uv.index(flower[j])
    return res

def yclassifier(dset, x):
    res = np.zeros((len(dset),1))
    uv = list(dset.iloc[:,-1].unique())
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
    I = X[0].shape[1]

    Gright = np.dot(np.reshape(tt[D], (R, I)),X[D].transpose())
    for i in [i for i in range(1,D)][::-1]:
        Gright = linalg.khatri_rao(Gright, X[i].transpose())
        Gright= np.reshape(tt[i], (R, R * I)).dot(Gright)
    superCore = linalg.khatri_rao(Gright, X[0].transpose())
    superCore = np.reshape(tt[0],(1,R*I)).dot(superCore).transpose()

    return superCore

def t_test(dset, I, iter, R, plot=False):
    train, test = train_test_split(dset,test_size=0.33)
    Xtrain = featurespace(train,I)
    # Xtest = featurespace(test,I)
    y = yspace(train)
    yc = yclassifier(train, 0)
    # ntt = initrandomtt(train,I,-10,10,R)
    accs = []
    ntt = np.asarray(ttest(data))
    for i in range(len(ntt)):
        print(ntt[i])


    for j in range(iter):
    # while (int(accuracy) < int(acc)):
        ntt = tt_ALS(ntt,Xtrain,yc,R)

        model = supercore(ntt, Xtrain)
        # print(f'model = {model}')
        #compare
        count = 0
        for i in range(len(model)):
            # print(y[i], yc[i], model[i], (model[i] * yy[i]) >0, (model[i] * yy[i]))
            if model[i] * yc[i] >= 0:
                count += 1
                # print(np.round(model[i],1), model[i], y[i])
        accuracy = np.round((count/len(model)) * 100,2)
        accs.append(accuracy)
        print(f'At iteration {j+1}, accuracy = {accuracy}')
    if plot == True:
        plt.plot(accs)
        plt.show()
    print(f'accuracy is: {accuracy}, it took {iter} iterations')

def ppinv(M):
    return np.linalg.inv(M.T*M) * M.T

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


if __name__ == "__main__":
    #datasets:
    iris = '/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv'
    indiaan = "/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/pima-indians-diabetes.csv"
    wine = "/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/winequality-white.csv"

    #variables
    I = 3 #nauwkeurigheid
    R = 4 #Rank
    feature = 0
    iter = 10
    dataset = iris #iris #indiaan #wine

    data = datareader(dataset)

    t_test(data, I, iter, R)
    #Metrics
    print(time.process_time())



"""
voorwaarden dataset:
- Laatste column is soort, mag ook een getal zijn
- enkel numerieke waarden in overige columns
- Accepteert comma en puntcomma seperated files
- meer rijen dan columns
- effectiever met grotere datasets

"""