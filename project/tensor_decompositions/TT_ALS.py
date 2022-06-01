import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.model_selection import train_test_split

iris = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv')
dataset = iris.iloc[:,[1,2,3,4,5]]
# print(dataset)

def initrandomtt(J, r):
    res = np.array([np.random.rand(r,1,J),np.random.rand(r,r,J),np.random.rand(r,r,J),np.random.rand(1,r,J)],dtype=object)

    return res

def rightSuperCore(tt,X,d):
    R = 2
    I = X[d].shape[1]
    D = len(tt)-1
    L = 1
    N = X[d].shape[0]

    D = len(tt)-1
    a = np.arange(d+2, D+1, 1).tolist()[::-1]

    # print(f'tt[{D}] = {tt[D].shape}')
    Gright = np.dot(np.reshape(tt[D],(R,J)),X[D].transpose())            #X4                         # eerste
    print(f'a = {d, a}')
    for i in a:

        Gright = linalg.khatri_rao(Gright,X[i-1].transpose())         #Xi       # Gright = np.kron(np.reshape(Gright.transpose(),(1,300)),X) # Tweede
        Gright = np.reshape(tt[i-1],(R,R*I)).dot(Gright)
        # Gright = np.reshape(tt[i-1], ((tt[i-1]).shape[1],np.prod((tt[i-1].shape[1:3])))).dot(Gright)        # laatste


    if d ==0:

        return Gright
    else:
        Gright = linalg.khatri_rao(Gright,X[d].transpose())  #X[d]
        # print(f'right = {Gright.shape, }')
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
        Gleft = Gleft.dot(X[d].transpose())

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
        Ri2, Ri1, J = (tt[i+1]).shape

        temp = np.reshape(tt[i+1],(I*R, R))  # Ri1 x J x Ri2 -> JRi1 x Ri2

        Gleft = Gleft.dot(temp)  # N x LRi2
        Gleft = np.reshape(Gleft, (N, L * R))  # NL x Ri2 -> N x LRi2

    if d == D:
        # print(f'Gleft = {Gleft.shape, X.shape, N,J,L}')
        # print(linalg.khatri_rao(Gleft.transpose(), X.transpose()).shape)
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
        Gright = rightSuperCore(tt, X, d) # R2 x N
        Gleft = np.ones((L,L)) #L x L
        superCore = linalg.khatri_rao(Gright,X[0].transpose())     #X1
        # print(superCore.shape)
        return np.reshape(superCore, (N*L, I*R)) # NL x LR2

    elif d == 1:
        Gleft = leftSuperCore(tt, X, d)  # L x R2
        Gright = rightSuperCore(tt, X, d).transpose()  # J R3 x N
        # print(f'Gleft.shape = {Gleft.shape}, Gright.shape = {Gright.shape}')

        superCore = linalg.khatri_rao(Gright, Gleft.transpose()) # JR3 x N kron R2 x L -> R2JR3 x LN
        # print(f'shape = {(N,L, R2*JR3)}')
        return np.reshape(superCore, (N, R*I*R))  # NL x R2JR3

    elif d==D:
        Gleft = leftSuperCore(tt, X, d)  # N x JLRd
        # print(Gleft.shape, N, J*L*Rd)
        return np.reshape(Gleft, (N, J*R))  # NL x J*Rd

    else:
        Gright = rightSuperCore(tt, X, d)  # Rd1 x N
        Gleft = leftSuperCore(tt, X, d)  # N x JLRd
        # print(Gright.shape, Gleft.shape, [Rd1, N], [N,J * L * Rd])

    # superCore = np.kron((Gright.flatten()),Gleft) # N x L Rd J Rd1
    superCore = linalg.khatri_rao(Gright.transpose(),Gleft.transpose())

    return np.reshape(superCore, (N * L, R * J * R)) # N L x Rd J Rd1

def updateCore(tt,mpt0,X,d,y):
    if isinstance(y, np.ndarray):
        # print(y.shape)
        L = y.shape[0]
        # print(f'L = {L}')
    else:
        L = 1
    # print(f'update {d}')
    U = getUL(tt,X,d,L)                           #case d=1: NL x JLR2, case d!=1 Nl x RdJRd1
  # case d=1: JLR2, case d!=1 RdJRd1
    y = np.reshape(y, (100,1))
    UTy = U.transpose().dot(y)

    # UTy = linalg.khatri_rao(U.transpose(),y)
    UTU = U.transpose().dot(U)                                         #case d=1: JLR2x JLR2, case d!=1 RdJRd1 RdJRd1
    # print(UTU.dot(UTy).shape)
    w = np.linalg.pinv(UTU).dot(UTy)
    # print(f'pi) = {w.shape}')
    print(f'd = {d}: U.shape = {U.shape}, y.shape = {y.shape}, UTy.shape = {UTy.shape}, UTU.shape = {UTU.shape}, w.shape = {w.shape} ')
    return w

def tt_ALS(tt,X,y,iter):
    D= len(tt)-1
    mpt = tt.copy()
    swipe = [0,1,2,3,2,1]
    # print(swipe)
    dims = []
    for i in range(len(tt)):
        dims.append(tt[i].shape)

    for i in range(0,iter):
        for j in range(len(swipe)):
            d = swipe[j]
            # print(f'{i, j}:d = {d}')
            # print(i, j)
            newCore = updateCore(tt,mpt,X,d,y)
            mpt[d] = np.reshape(newCore,dims[d])
            # print(f'in iteration{i,j}, tt[{d}].shape = {tt[d].shape}, mpt[{d}].shape = {mpt[d].shape}')

    return mpt

def flat(array):
    res = []
    a = array.shape[0]
    # print(array[0].shape)
    for i in range(array.shape[0]):
        for j in range(array[i].shape[0]):
            for k in range(array[i].shape[1]):
                for l in range(array[i].shape[2]):
                    res.append(array[i][j][k][l])
    return res

def featurespace1(dataset, feature, p):
    cnames = dataset.columns.values
    res = np.zeros((len(dataset),p))


    flower = dataset.iloc[:,feature]
    flower = list(flower)
    for i in cnames:
        for j in range(len(flower)):

            X = flower[j]
            Y = np.array([X**i for i in range(p)])
            res[j] = Y
    return res

def featurespace(dataset, p):
    cnames = dataset.columns.values
    res = [[[0 for _ in range(len(cnames))] for _ in range(len(dataset))] for _ in range(p)]


    for i in range(len(cnames)-1):
        for j in range(len(dataset)):
            print(i,j)
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
    # res = np.zeros(len(dataset))
    flower = list(dset.iloc[:, -1])

    for j in range(len(flower)):
        k = np.zeros(len(uv))
        k[uv.index(flower[j])] = 1
        res[j] = k

    return res

def yspace(dset):
    res = np.zeros((len(dset),1))
    uv = list(dataset.iloc[:,-1].unique())
    # res = np.zeros(len(dataset))
    flower = list(dset.iloc[:, -1])
    for j in range(len(flower)):
        res[j] = uv.index(flower[j])
    return res

#variables:
J = 4 #nauwkeurigheid
feature = 0
iter = 3

train, test = hussel(dataset)
X = featurespace(train,J)
# print(featurespace1(dataset,feature,J))

y = yspace(train)

# print(y)
tt = initrandomtt(J, r=2)



print(tt_ALS(tt,X,y,iter))
print(f'oud = {tt}')

# print(flat(y))

# Gright = reshape(tt.cores[i-1],(size(tt.cores[i-1])[1],prod(size(tt.cores[i-1])[2:3])))*Gright'