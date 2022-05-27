import numpy as np
import pandas as pd
from scipy import linalg

iris = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv')
dataset = iris.iloc[:,[1,2,3,4]]

def initrandomtt(J=4, r=2):
    res = np.array([np.random.rand(r,1,J),np.random.rand(r,r,J),np.random.rand(r,r,J),np.random.rand(1,r,J)],dtype=object)

    return res

def rightSuperCore(tt,X,d):

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
        # print(f'right = {Gright.shape, }')
    return (Gright.transpose())

def leftSuperCore(tt,X,d):

    D = len(tt)-1
    # print(f'D = {D}')

    r2,r1,L = tt[0].shape

    Gleft = np.reshape(tt[0],(L,r2))

    N = X.size
    if d == 1:
        return Gleft
    r3,r2,J = tt[1].shape
    Gleft = Gleft.dot(np.reshape(tt[2],(r2,J*r3)))
    Gleft = np.reshape(Gleft, (J,L*r3))
    Gleft = X.dot(Gleft)

    if d == 2:
        # print(f'Gleft = {Gleft.shape, L, r3}')
        return Gleft

    for i in range(1,d-1):
        # print(i)
        Ri1, Ri, J = tt[i].shape

        Gleft = linalg.khatri_rao(np.transpose(Gleft), X.transpose()).transpose()   # N x JLRi1
        N, JLRi1 = Gleft.shape
        L = JLRi1 // (J*Ri)

        Gleft = np.reshape(Gleft, (N*L, J*Ri))   # N x JLRi1 ->  NL x JRi1
        Ri2, Ri1, J = (tt[i+1]).shape

        temp = np.reshape(tt[i+1],(J*Ri1, Ri2))  # Ri1 x J x Ri2 -> JRi1 x Ri2

        Gleft = Gleft.dot(temp)  # N x LRi2
        Gleft = np.reshape(Gleft, (N, L * Ri2))  # NL x Ri2 -> N x LRi2

    if d == D:
        # print(f'Gleft = {Gleft.shape, X.shape, N,J,L}')
        # print(linalg.khatri_rao(Gleft.transpose(), X.transpose()).shape)
        return linalg.khatri_rao(Gleft.transpose(), X.transpose()).transpose()  # N x JLRd

    return Gleft  # N x LRd

def getUL(tt, X, d, L):
    D = len(tt)-1
    # print(L)
    N = X.shape[0]
    R2,R1,L = tt[0].shape
    Rd1,Rd,J = tt[d].shape
    # print(L)
    # print(R2,R1,L)
    # print(Rd1,Rd,J)

    # print(R1,L,R2,N)
    if d == 0:
        Gright = rightSuperCore(tt, X, d) # R2 x N
        Gleft = np.ones((L,L)) #L x L
        superCore = np.kron(Gright,Gleft) # R2 x N kron L x L -> LR2 x L N

        return np.reshape(superCore, (N*L, L*R2)) # NL x LR2

    elif d == 1:
        Gleft = leftSuperCore(tt, X, d)  # L x R2
        Gright = rightSuperCore(tt, X, d).transpose()  # J R3 x N
        JR3, N = Gright.shape
        superCore = np.kron(Gright, Gleft.transpose()) # JR3 x N kron R2 x L -> R2JR3 x LN
        # print(f'shape = {(N,L, R2*JR3)}')
        return np.reshape(superCore, (N*L, R2*JR3))  # NL x R2JR3

    elif d==D:
        Gleft = leftSuperCore(tt, X, d)  # N x JLRd
        # print(Gleft.shape, N, J*L*Rd)
        return np.reshape(Gleft, (N*L, J*Rd))  # NL x J*Rd

    else:
        Gright = rightSuperCore(tt, X, d)  # Rd1 x N
        Gleft = leftSuperCore(tt, X, d)  # N x JLRd
        # print(Gright.shape, Gleft.shape, [Rd1, N], [N,J * L * Rd])

    # superCore = np.kron((Gright.flatten()),Gleft) # N x L Rd J Rd1
    superCore = linalg.khatri_rao(Gright.transpose(),Gleft.transpose())

    return np.reshape(superCore, (N * L, Rd * J * Rd1)) # N L x Rd J Rd1

def updateCore(tt,mpt0,X,d,y):
    if isinstance(y, np.ndarray):
        # print(y.shape)
        L = y.shape[0]
        # print(f'L = {L}')
    else:
        L = 1
    # print(f'update {d}')
    U = getUL(tt,X,d,L)                           #case d=1: NL x JLR2, case d!=1 Nl x RdJRd1
    # y = (y[d]).flatten()
    # print(y[0])
    # UTy = U.dot(np.reshape(y, (U.shape[1],int(len(y)//U.shape[1]))))  # case d=1: JLR2, case d!=1 RdJRd1
    UTy = U.dot(y[d].flatten())
    UTU = U.transpose().dot(U)                                         #case d=1: JLR2x JLR2, case d!=1 RdJRd1 RdJRd1


    # Pnew = P0inv[d] +(UTU//s**2)                                        #case d=1: JLR2x JLR2, case d!=1 RdJRd1 RdJRd1
    # print(f'U.shape = {U.shape}, flat(y).shape = {len(y)}, UTy.shape = {UTy.shape}, UTU.shape = {UTU.shape}, Pnew.shape = {1}')
    # mnew = np.linalg.pinv(Pnew).dot(UTy)

    w = np.linalg.pinv(U).dot(UTy)
    # print(f'pi) = {w.shape}')
    return w

def tt_ALS(tt,X,y,iter):
    D= len(tt)-1
    print(D)
    mpt = tt.copy()
    swipe = [0,1,2,3,2,1]
    # print(swipe)
    dims = []
    for i in range(len(tt)):
        dims.append(tt[i].shape)
    print(dims)
    P = np.arange(0,D)
    for i in range(0,iter):
        for j in range(len(swipe)):
            d = swipe[j]
            # print(f'{i, j}:d = {d}')
            # print(i, j)
            newCore = updateCore(tt,mpt,X,d,y)
            mpt[d] = np.reshape(newCore,dims[d])
            # print(f'in iteration{i,j}, tt[{d}].shape = {tt[d].shape}, P[{d}].shape = {P[d].shape}')

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
#datasplit


def featurespace(dataset, feature, p):
    # dataset = list(dataset)
    print(1)
    res = np.zeros((len(dataset),p))
    # print(res.shape)
    flower = dataset.iloc[:,feature]
    for j in range(len(flower)):
        X = flower[j]
        print(X)
        Y = np.array([1, X, X**2, X**3])

        res[j] = Y
    return res


X = featurespace(dataset,1,4)
print(X)
tt = initrandomtt(J = 4, r = 2)
# print(X[0])

print(getUL(tt, X, 0, 3).shape)
# print(getUL(tt, X, 1, 3).shape)
# print(getUL(tt, X, 2, 3).shape)
# print(getUL(tt, X, 3, 3).shape)
y = tt.copy()
# P0inv = np.arange(0,8)

# print(y.type)

print(f'nieuw = {tt_ALS(tt,X,y,3)}')
print(f'oud = {tt}')

# print(flat(y))

# Gright = reshape(tt.cores[i-1],(size(tt.cores[i-1])[1],prod(size(tt.cores[i-1])[2:3])))*Gright'