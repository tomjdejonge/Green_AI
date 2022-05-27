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

def getUL(tt, X, d):
        D = len(tt)-1
        N = X.shape[0]
        R2,R1,L = tt[0].shape
        Rd1,Rd,J = tt[d].shape
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
            Gright = rightSuperCore(tt, X, d)  # J R3 x N
            JR3, N = Gright.shape
            superCore = np.kron(Gright, Gleft.transpose()) # JR3 x N kron R2 x L -> R2JR3 x LN

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


#datasplit

X = featurespace(dataset,1,4)
tt = initrandomtt(I = 4, r = 2)

getUL(tt, X, 0)
getUL(tt, X, 1)
getUL(tt, X, 2)
getUL(tt, X, 3)

# Gright = reshape(tt.cores[i-1],(size(tt.cores[i-1])[1],prod(size(tt.cores[i-1])[2:3])))*Gright'