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
        # print(linalg.khatri_rao(X[i-1].transpose(), Gright) == linalg.khatri_rao(Gright,X[i-1].transpose()))
        Gright = linalg.khatri_rao(Gright, X[i-1].transpose())
        Gright = np.reshape(tt[i-1], (R, R*I)).dot(Gright)
        # print(f'gright = {Gright}')

    Gright = linalg.khatri_rao(Gright, X[d].transpose())

    return Gright
"""  if d ==0:
        Gright = linalg.khatri_rao(X[2].transpose(), Gright)
        Gright = np.reshape(tt[2], (R, R * I)).dot(Gright)
        Gright = linalg.khatri_rao(X[1].transpose(), Gright)
        Gright = np.reshape(tt[1], (R, R * I)).dot(Gright)
        Gright = linalg.khatri_rao(X[0].transpose(), Gright)
        return Gright"""

def leftSuperCore(tt,X,d):
    R = 2
    I = X[d].shape[1]
    D = len(tt)-1
    L = 1
    N = X[d].shape[0]


    Gleft = np.reshape(tt[0],(R,I)).dot(X[0].transpose())
    if d == 1:
        return Gleft
    print(d, Gleft.shape, X[d].shape)
    for i in range(1,d):
        print(d, i, Gleft.shape, X[d].shape)

        Gleft = linalg.khatri_rao(Gleft, X[i].transpose())
        Gleft = np.reshape(Gleft, (N, I*R)).dot(np.reshape(tt[i],(I*R, R)))
        Gleft = np.reshape(Gleft, (min(Gleft.shape),max(Gleft.shape)))

    if d ==D:
        Gleft = linalg.khatri_rao(Gleft,X[D].transpose())

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

    Gleft = np.reshape(Gleft, (min(Gleft.shape), max(Gleft.shape)))
    Gright = np.reshape(Gright, (min(Gright.shape), max(Gright.shape)))
    superCore = linalg.khatri_rao(Gright, Gleft)

    return np.reshape(superCore, (N, R * I * R))  # NL x R2JR3


def updateCore(tt,X,d,y):

    U = getUL(tt,X,d)
    UTy = U.transpose().dot(np.reshape(y, (max(y.shape),1)))

    UTU = U.transpose().dot(U)

    w = linalg.inv(UTU).dot(UTy)

    #np.linalg.inv(M.T*M) * M.T
    print(f'd = {d}: U.shape = {U.shape}, y.shape = {y.shape}, UTy.shape = {UTy.shape}, UTU.shape = {UTU.shape}, w.shape = {w.shape}, UTU[0][0] = {U[0][0]} ')
    return w

def tt_ALS(tt,X,y,iter):
    D= len(tt)-1
    xss = ([[i for i in range(D+1)], [i for i in range(1,D)][::-1]])
    swipe = [x for xs in xss for x in xs]
    dims = []

     # [collect(1:D)..., collect(D-1:-1:2)...] = [0,1,2,3,2,1]??

    for i in range(len(tt)):
        dims.append(tt[i].shape)

    for i in range(0,iter):
        for j in range(len(swipe)):
            d = swipe[j]

            newCore = updateCore(tt,X,d,y)

            tt[d] = np.reshape(newCore,dims[d])
            # print(tt)

    return tt

def featurespace(dataset, p):
    cnames = dataset.columns.values
    res = [[[0 for _ in range(p)] for _ in range(len(dataset))] for _ in range(len(cnames))]
    # print(len(cnames), len(dataset), p)
    for i in range(len(cnames)-1):
        flower = list(dataset.iloc[:, i])
        for j in range(len(dataset)):
            # print(i,j)
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
    for i in [i for i in range(1,D)][::-1]:
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
    # print('yy',yy)
    # print(y)
    tt = initrandomtt(dset, I,0,3,r=2)
    traintt = tt_ALS(tt,Xtrain,yy,iter)

    model = supercore(traintt, Xtrain)
    # print(f'model = {model}')
    #compare
    count = 0
    for i in range(len(model)):
        if model[i] * yy[i] > 0:
            count += 1
            # print(np.round(model[i],1), model[i], y[i])
    acc = (count/len(model)) * 100
    print(f'accuracy is: {acc}')

def ppinv(M):
    return np.linalg.inv(M.T*M) * M.T

def seper(file):
    df_comma = pd.read_csv(file, nrows=1, sep=",")
    df_semi = pd.read_csv(file, nrows=1, sep=";")
    if df_comma.shape[1] > df_semi.shape[1]:
        print(",")
    else:
        print(";")

iris = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv')
# iris = iris.iloc[:,[1,2,3,4,5]]
# penguins = pd.read_csv("/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/penguins.csv")
indiaan = pd.read_csv("/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/pima-indians-diabetes.csv")
# print(indiaan)
# seper('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv')

# indiaandataset = iris.iloc[:,[1,2,3,4,5,6,7]]
# print(indiaandataset)


#variables:
I = 4 #nauwkeurigheid
feature = 0
iter = 1
dataset = iris

t_test(dataset, I, iter)


print(time.process_time())
# model = Sequential(t_test(irisdataset, I, iter))
# model.add(Dense(8, activation='softmax'))
# print(get_flops(model))


"""[2022/05/23 16:19] Eva Memmel
function testeva(tt::MPT, X::Matrix, y::Array, P0inv::Vector{Array{Float64}},σ::Float64,maxiter::Int64)

    #y: matrix y(:,k) contains kth output
    #u: matrix u(:,k) contains kth input
    #M: integer, memory of each Volterra kernel
    #rnks: matrix, contains r_2 ... r_{D-1}
    #minimax: min and max value for uniform distribution in prior
    #σ: initial variance
    
    
    
        #-----------------------------------------------------------
        # TT_ALS in Bayesian framework
        #
        # Eva September 2021
        #----------------------------------------------------------- 
        D = order(tt)
        coreDims = coreDimensions(tt)
        swipe = [collect(1:D)..., collect(D-1:-1:2)...]
        mpt0 = deepcopy(tt)
        P = Vector{Array{Float64}}(undef,D)
        for i = 1:maxiter
            for j = 1:2D-2
                d = swipe[j]
                newCore, Pnew = updateCoreL(tt,mpt0,X,y,P0inv,d,σ) # case d=1: JLR2, case d!=1 RdJRd1
                tt.cores[d] = reshape(newCore,(coreDims[d]))
                P[d] = Pnew
            end      
        end
        #take dimension L out of first core
        L = size(y,2)
        return tt, P
    end

[2022/05/23 16:19] Eva Memmel
    function rightSuperCoreL(tt::MPT, X::Matrix, d::Int64)
        D = length(tt.cores)
        Gright = reshape(tt.cores[D],size(tt.cores[D])[1:2])*X' # RD x I * I  N -> RD x N 
        for i = D:-1:d+2
            Gright = dotkron(collect(Gright'),X) #N x JRi
            Gright = reshape(tt.cores[i-1],(size(tt.cores[i-1])[1],prod(size(tt.cores[i-1])[2:3])))*Gright' #Ri-1 x JRi * JRi x N -> Ri-1 x N
        end
        if d ==1
            return Gright #R2 x N
        end
        Gright = dotkron(collect(Gright'),X) #N x JR3
        return collect(Gright')
    end

[2022/05/23 16:20] Eva Memmel
    function leftSuperCoreL(tt::MPT, X::Matrix, d::Int64)
        D = order(tt)
        r1,L,r2 = size(tt.cores[1])
        Gleft = reshape(tt.cores[1],L,r2)
        N = size(X,1)
        if d == 2
            return Gleft # L x r2
        end
        r2,J,r3 = size(tt.cores[2])
        Gleft = Gleft * reshape(tt.cores[2],r2,J*r3) #L x Ir3
        Gleft = reshape(permutedims(reshape(Gleft,L,J,r3),[2,1,3]),J,L*r3) #I x Lr3
        Gleft = X*Gleft # N x LR3
        if d == 3
            return Gleft # N x LR3
        end
        for i = 2:d-2
            Ri,J,Ri1 = size(tt.cores[i])
            Gleft = dotkron(Gleft,X) # N x JLRi1
            Gleft = reshape(permutedims(reshape(Gleft,(N,J,L,Ri1)),(1, 3, 2, 4)),(N*L,J*Ri1)) # N x JLRi1 ->  NL x JRi1
            Ri1,J,Ri2 = size(tt.cores[i+1])
            temp = reshape(permutedims(tt.cores[i+1],(2,1,3)),(J*Ri1,Ri2)) # Ri1 x J x Ri2 -> JRi1 x Ri2
            Gleft = Gleft*temp #N x LRi2
            Gleft = reshape(Gleft,(N,L*Ri2)) #NL x Ri2 -> N x LRi2
        end
        if d == D
            return dotkron(Gleft,X) # N x JLRd
        end
        return Gleft # N x LRd
    end

[2022/05/23 16:20] Eva Memmel
    function getUL(tt::MPT, X::Matrix, d::Int64, L::Int64)
        D = length(tt.cores)
        N = size(X)[1]    
        R1,L,R2 = size(tt.cores[1])
        Rd,J,Rd1 = size(tt.cores[d])
        if d == 1
            Gright = rightSuperCoreL(tt, X, d) # R2 x N
            Gleft = Matrix(1.0I,L,L) #L x L 
            superCore = kron(Gright,Gleft) # R2 x N kron L x L -> LR2 x L N
            superCore = reshape(permutedims(reshape(superCore,L,R2,L,N),[4 3 1 2]),N*L,L*R2) # NL x LR2
            return superCore  # NL x LR2 
        elseif d == 2   
            Gleft = leftSuperCoreL(tt,X,d) # L x R2
            Gright = rightSuperCoreL(tt,X,d) # J R3 x N
            superCore = kron(Gright,Gleft') # JR3 x N kron R2 x L -> R2JR3 x LN
            superCore = reshape(permutedims(reshape(superCore,Rd,J,Rd1,L,N),[5 4 1 2 3]),N*L,Rd*J*Rd1) # NL x R2JR3
            return superCore
        elseif d == D 
            Gleft = leftSuperCoreL(tt,X,d) # N x JLRd        
            return reshape(permutedims(reshape(Gleft,(N,J,L,Rd)),(1,3,2,4)),(N*L,J*Rd)) #NL x J*Rd
        else    
            Gright = rightSuperCoreL(tt, X, d) # Rd1 x N
            Gleft = leftSuperCoreL(tt,X,d) # N x JLRd
        end
        superCore = dotkron(collect(Gright'),Gleft) # N x L Rd J Rd1 
        superCore = reshape(reshape(superCore,N,L,Rd,J,Rd1),N*L,Rd*J*Rd1) # N L x Rd J Rd1
        return superCore #N L x Rd J Rd1
    end

[2022/05/23 16:20] Eva Memmel
    function updateCoreL(tt::MPT, mpt0::MPT, X::Matrix, y::Array, P0inv::Vector{Array{Float64}}, d::Int64,σ::Float64)
        #-----------------------------------------------------------
        # Bayesian updating of one core while fixing all others in Volterra Setting
        #
        # Eva Memmel, Oktober 2021
        #-----------------------------------------------------------
        if isa(y,Matrix)
            L = size(y,2); #L: number of outputs
        else
            L = 1
        end
        U = getUL(tt,X,d,L); #case d=1: NL x JLR2, case d!=1 Nl x RdJRd1 
        UTy = U'*vec(y); # case d=1: JLR2, case d!=1 RdJRd1 
        UTU = U'U; #case d=1: JLR2x JLR2, case d!=1 RdJRd1 RdJRd1 
      
        m0 = vec(mpt0.cores[d])  # case d=1: JLR2, case d!=1 RdJRd1
        Pnew = P0inv[d] + UTU/(σ^2) #case d=1: JLR2x JLR2, case d!=1 RdJRd1 RdJRd1
        mnew = pinv(Pnew)*(UTy/(σ^2) + P0inv[d]*m0)  
        return mnew, Pnew
    end

"""