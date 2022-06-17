import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.model_selection import train_test_split
import time as time
from matplotlib import pyplot as plt
import pdb


# from svdtest import ttest
def dotkron(A, B):
    N, DA = A.shape
    N, DB = B.shape
    temp = np.ones((N, DA * DB))
    for n in range(N):
        # print(n)
        temp[n, :] = np.kron(A[n, :], B[n, :]).transpose()
        # print(np.kron(A[n,:],B[n,:]))
    return temp

def ppinv(M):
    return np.linalg.inv(M.transpose().dot(M)).dot(M.T)


def initrandomtt(dataset, J, min, max, r):
    start = [np.random.randint(min, max, size=(r, 1, J))]
    for i in range(len(dataset.columns.values) - 3):
        start.append(np.array([np.random.randint(min, max, size=(r, r, J))]))
    start.append(np.array([np.random.randint(min, max, size=(1, r, J))]))

    return np.array(start, dtype=object)


def rightSuperCore(tt, X, d, R):
    N = X[d].shape[0]
    D = len(tt) - 1

    I = tt[D].shape[-1]

    # reshape, contract last core
    Gright = np.dot(np.reshape(tt[D], (R, I)), X[D].transpose())  # Rd x N

    for i in np.arange(d + 1, D, 1).tolist()[::-1]:
        # dotkron reshape contract middle cores
        Gright = dotkron(Gright.transpose(), X[i])  # N x I Ri

        Gright = np.reshape(tt[i], (R, R * I)).dot(Gright.transpose())  # Ri-1 x JRi * JRi x N -> Ri-1 x N

    # last dotkron

    Gright = dotkron(Gright.transpose(), X[d])

    return Gright.transpose()  # I R x N


"""  if d ==0:
        Gright = linalg.khatri_rao(X[2].transpose(), Gright)
        Gright = np.reshape(tt[2], (R, R * I)).dot(Gright)
        Gright = linalg.khatri_rao(X[1].transpose(), Gright)
        Gright = np.reshape(tt[1], (R, R * I)).dot(Gright)
        Gright = linalg.khatri_rao(X[0].transpose(), Gright)
        return Gright"""


def leftSuperCore(tt, X, d, R):
    I = X[d].shape[1]
    D = len(tt) - 1
    L = 1
    N = X[d].shape[0]

    # reshape first core, contract
    Gleft = (X[0]) @ np.reshape(tt[0], (I, R))  # N x R
    if d == 1:
        return Gleft
    for i in range(1, d):
        # dotkron, contract, reshape for middle cores

        Gleft = dotkron(Gleft, X[i])  # N x I R

        tt[i] = tt[i].reshape(R, R, I)
        temp = np.transpose(tt[i], (2, 0, 1))  # I x R x R

        temp = np.reshape(temp, (I * R, R))  # I R x R
        Gleft = Gleft @ temp  # N x R

    if d == D:
        # only last dotkron if last core
        Gleft = dotkron(Gleft, X[D])

    return Gleft  # N x I R


def getUL(tt, X, d, R):
    I = X[d].shape[1]
    D = len(tt) - 1
    N = X[d].shape[0]
    # combine both supercores, only with khatri rao if not first or last
    if d == 0:
        superCore = rightSuperCore(tt, X, d, R)  # R2 x N

        return np.reshape(superCore, (N, I * R))  # NL x LR2

    elif d == D:
        Gleft = leftSuperCore(tt, X, d, R)  # N x  I Rd
        Gleft = np.reshape(Gleft, (N, I, R))
        Gleft = np.transpose(Gleft, (0, 2, 1))
        Gleft = np.reshape(Gleft, (N, R * I))

        return np.reshape(Gleft, (N, I * R))

    Gright = rightSuperCore(tt, X, d, R)  # I Rd1 x N
    Gleft = leftSuperCore(tt, X, d, R)  # N x LRd

    superCore = dotkron(Gright.transpose(), Gleft)

    return superCore  # NL x R2JR3


def updateCore(tt, X, d, y, R):
    U = getUL(tt, X, d, R)

    # UTy = U.transpose()@(np.reshape(y, (max(y.shape),1)))
    #
    # UTU = U.transpose()@(U)

    # choose one, linalg.pinv(U).dot(y) gives singular matrix error
    w = linalg.pinv(U).dot(y)
    # w = linalg.pinv(UTU)@(UTy)

    # np.linalg.inv(M.T*M) * M.T
    # print(f'd = {d}: U.shape = {U.shape}, y.shape = {y.shape}, UTy.shape = {UTy.shape}, UTU.shape = {UTU.shape}, w.shape = {w.shape}, UTU[0][0] = {U[0][0]} ')
    return w


def tt_ALS(tt, X, y, R):
    D = len(tt) - 1
    xss = ([[i for i in range(D + 1)], [i for i in range(1, D)][
                                       ::-1]])  # ([[i for i in range(D+1)], [i for i in range(1,D)][::-1]])  #([[i for i in range(1,D)][::-1],[i for i in range(D+1)]])
    swipe = [x for xs in xss for x in xs]
    dims = []

    # [collect(1:D)..., collect(D-1:-1:2)...] = [0,1,2,3,2,1]??

    for i in range(len(tt)):
        dims.append(tt[i].shape)

    # iterate over the swipe
    for j in range(len(swipe)):
        d = swipe[j]
        newCore = updateCore(tt, X, d, y, R)
        tt[d] = np.reshape(newCore, dims[d])
        # print(tt)
    return tt


def featurespace(dtset, p):
    cnames = dtset.columns.values
    minv = np.min(dtset)
    maxv = np.max(dtset)
    # print(minv, maxv)

    res = [[[0 for _ in range(p)] for _ in range(len(dtset))] for _ in range(len(cnames) - 1)]
    for i in range(len(cnames) - 1):
        flower = list(dtset.iloc[:, i])
        fmin = min(flower)
        fmax = max(flower)

        for j in range(len(dtset)):
            res[i][j] = [(((flower[j] - fmin) / (fmax - fmin)) * 10) ** i for i in range(p)]

    # normalize the feature space
    # (res - minv) / (maxv - minv)
    return np.asarray(res)


def yspace(dset):
    res = np.zeros((len(dset), 1))
    uv = list(dset.iloc[:, -1].unique())
    flower = list(dset.iloc[:, -1])
    for j in range(len(flower)):
        res[j] = uv.index(flower[j])
    return res


# yclassifier, xth flower gets assigned 1, rest -1
def yclassifier(dset, x):
    res = np.zeros((len(dset), 1))
    uv = list(dset.iloc[:, -1].unique())
    flower = list(dset.iloc[:, -1])

    for j in range(len(flower)):
        if uv.index(flower[j]) == x:
            res[j] = 1
        else:
            res[j] = -1
    return res


# contract the weights (tt) with the values (X) to test
def predict(tt, X, R, d=0):
    # contract
    N = X[d].shape[0]
    D = len(tt) - 1

    I = tt[D].shape[-1]

    # reshape, contract last core
    Gright = np.dot(np.reshape(tt[D], (R, I)), X[D].transpose())  # Rd x N

    for i in np.arange(d + 1, D, 1).tolist()[::-1]:
        # dotkron reshape contract middle cores
        Gright = dotkron(Gright.transpose(), X[i])  # N x I Ri

        Gright = np.reshape(tt[i], (R, R * I)).dot(Gright.transpose())  # Ri-1 x JRi * JRi x N -> Ri-1 x N

    Gright = dotkron(Gright.transpose(), X[0])

    Gright = np.reshape(tt[0], (1, R * I)).dot(Gright.transpose())

    return Gright.transpose()

def naive(file, split, p):
    start = time.process_time()
    train, test = train_test_split(file, test_size=split)
    cnames = train.columns.values

    y = yclassifier(train, p)
    # print(minv, maxv)

    res = [[0 for _ in range(p)] for _ in range(len(train))]
    for j in range(len(train)):
        flower = list(train.iloc[j, :-1])
        res[j] = [(flower[0]) ** k for k in range(p)]
        for i in range(1,len(cnames) - 1):
            print(j,i, res[j])
            res[j] = np.outer(res[j],[(flower[i]) ** k for k in range(p)])

    res = np.reshape(np.asarray(res), (len(train), (p)**(len(cnames)-1)))
    print(res.shape)

    w = linalg.pinv(res).dot(y)
    yhat = res.dot(w)
    count = 0
    for i in range(len(yhat)):

        if yhat[i] * y[i] >= 0:
            count += 1
            # print(yhat[i], y[i])

    accuracy = np.round((count / len(yhat)) * 100, 2)
    stop = time.process_time()

    print(f'time = {stop-start}, accuracy = {accuracy}')

# function to test
def t_test(dset, I, iter, R, plot=False):  # Predicting
    start = time.process_time()
    train, test = train_test_split(dset, test_size=0.33)

    Xtrain = featurespace(train, I)
    # print(Xtrain)
    # Xtest = featurespace(test,I)
    # y = yspace(train)
    yc = yclassifier(train, 0)  # labels
    ntt = initrandomtt(train, I, 0, 5, R)
    accs = []
    # ntt = np.asarray(ttest(data))
    naive(dset, 0.33, I)


    for j in range(iter):
        # while (int(accuracy) < int(acc)):
        ntt = tt_ALS(ntt, Xtrain, yc, R)

        model = predict(ntt, Xtrain, R)
        # print(f'model = {model}')
        # compare
        count = 0
        for i in range(len(model)):
            # print(y[i], yc[i], model[i], (model[i] * yy[i]) >0, (model[i] * yy[i]))
            if model[i] * yc[i] >= 0:
                count += 1
                # print(np.round(model[i],1), model[i], y[i])
        accuracy = np.round((count / len(model)) * 100, 2)
        accs.append(accuracy)
        print(f'At iteration {j + 1}, accuracy = {accuracy}')
    if plot == True:
        plt.plot(accs)
        plt.show()
    print(f'accuracy is: {accuracy}, it took {iter} iterations')
    stop = time.process_time()
    return (stop-start)


def datareader(location):
    df_comma = pd.read_csv(location, nrows=1, sep=",")
    df_semi = pd.read_csv(location, nrows=1, sep=";")
    if df_comma.shape[1] > df_semi.shape[1]:
        sepp = ','
    else:
        sepp = ';'
    dframe = pd.read_csv(location, sep=sepp)

    if len(dframe.iloc[:, 0]) == dframe.iloc[-1, 0] or len(dframe.iloc[:, 0]) == dframe.iloc[-1, 0] - 1:
        dframe.drop(columns=dframe.columns[0],
                    axis=1,
                    inplace=True)
    return dframe




def recommend(dset):
    iterations = np.arange(1,20)
    times = []
    for ep in iterations:

        print(f'calculating ep {ep}')

        times.append(t_test(dset, 4, ep, 4))

    return iterations, times


if __name__ == "__main__":
    # datasets:
    iris = "C:/Users/tommo/Downloads/iris.csv"
    indiaan = "C:/Users/tommo/Downloads/pima-indians-diabetes.csv"
    wine = "C:/Users/tommo/Downloads/winequality-white.csv"

    # variables
    I = 3  # nauwkeurigheid
    R = 5  # Rank
    feature = 0
    iter = 5
    dataset = iris  # iris #indiaan #wine

    data = datareader(dataset)
    # iterations, times = recommend(data)
    # plt.plot(iterations, times)
    # plt.show()
    t_test(data, I, iter, R)
    # print(recommend(data))
    # Metrics


"""
voorwaarden dataset:
- Laatste column is soort, mag ook een getal zijn
- enkel numerieke waarden in overige columns
- Accepteert comma en puntcomma seperated files
- meer rijen dan columns
- effectiever met grotere datasets

"""