
from TT_SVD import tensortrain, tt_reconstruction
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty

img = Image.open('images/dog.jpg')
img2 = Image.open('images/baboon.png')
core, d, r, n = tensortrain(img)
B = tt_reconstruction(core, d, r, n)
# print(B.shape)

A = np.array([[1,3,5],[2,4,6]])
Y = np.array([[[1,4,7,10],[2,5,8,11],[3,6,9,12]],
                [[13,16,19,22],[14,17,20,23],[15,18,21,24]]])
V = np.array([[1,2,3,4]])
# print(V)
# Y = np.reshape(Y, (3,4,2))
# A = A.transpose(1,0)
# print(Y.shape)

def unfold(tensor, x):
    if x==1:
        tensor = tensor.transpose(1, 0, 2)
        shape = tensor.shape
        res = np.reshape(tensor, (shape[0],shape[1]*shape[2]))
    elif x==2:
        tensor = tensor.transpose(2,0,1)
        shape = tensor.shape
        res = np.reshape(tensor, (shape[0],shape[1]*shape[2]))
    elif x==3:
        tensor = tensor.transpose(0, 2, 1)
        shape = tensor.shape
        res = np.reshape(tensor, (shape[0], shape[1] * shape[2]))
    return res  # tl.reshape(tl.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))
# https://www.kolda.net/publication/TensorReview.pdf


def nmultiplication(tensor, matrix, n):
    tshape = tensor.shape
    mshape = matrix.shape
    # print(tshape,"aaa",mshape)
    res = matrix.dot(unfold(tensor,n))
    return np.reshape(res,(mshape[0],tshape[0],tshape[2]))

def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return ty.moveaxis(ty.reshape(unfolded_tensor, full_shape), 0, mode)


def unfold(tensor, mode):
    return ty.reshape(ty.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def mode_dot(tensor, matrix_or_vector, mode):
    # the mode along which to fold might decrease if we take product with a vector
    mode = mode-1
    fold_mode = mode
    new_shape = list(tensor.shape)
    if len(matrix_or_vector.shape) == 2:  # Tensor times matrix
        # Test for the validity of the operation
        dim = 1
        new_shape[mode] = matrix_or_vector.shape[0]
        vec = False

    elif len(matrix_or_vector.shape) == 1:  # Tensor times vector
        if matrix_or_vector.shape[0] != tensor.shape[mode]:
            raise ValueError(
                'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                ))
        if len(new_shape) > 1:
            new_shape.pop(mode)
        else:
            # Ideally this should be (), i.e. order-0 tensors
            # MXNet currently doesn't support this though..
            new_shape = []
        vec = True
    else:
        raise ValueError
    res = np.dot(matrix_or_vector, unfold(tensor, mode))

    if vec:  # We contracted with a vector, leading to a vector
        return ty.reshape(res, shape=new_shape)
    else:  # tensor times vec: refold the unfolding
        return fold(res, fold_mode, new_shape)

# print(mode_dot(Y, A, 1))

# X = A.dot(unfold(Y,1))
# print(np.reshape(X,(2,2,4)).shape)


# W = V.dot(unfold(Y,2))
# print(np.reshape(W,(3,2)))
# print(nmultiplication(np.array(B).reshape(512,3,512),A,1))
# #nmodeproduct
# print(unfold(Y,1).dot(A.transpose(1,0)))
# print(ty.tenalg.mode_dot(Y,A,1))
# print(ty.tenalg.mode_dot(Y,A,1))
#
#
# print(np.matmul(unfold(Y,1),(A)))
# print(Y.shape, A.shape)
# print(unfold(Y,1).shape)
#
# print(unfold(Y,1).dot(A))
print(nmultiplication(Y, A, 1))

[Monday 16:19] Eva Memmel
function testeva(tt::MPT, X::Matrix, y::Array, P0inv::Vector{Array{Float64}},σ::Float64,maxiter::Int64)    #y: matrix y(:,k) contains kth output    #u: matrix u(:,k) contains kth input    #M: integer, memory of each Volterra kernel    #rnks: matrix, contains r_2 ... r_{D-1}    #minimax: min and max value for uniform distribution in prior    #σ: initial variance                    #-----------------------------------------------------------        # TT_ALS in Bayesian framework        #        # Eva September 2021        #-----------------------------------------------------------         D = order(tt)        coreDims = coreDimensions(tt)        swipe = [collect(1:D)..., collect(D-1:-1:2)...]        mpt0 = deepcopy(tt)        P = Vector{Array{Float64}}(undef,D)        for i = 1:maxiter            for j = 1:2D-2                d = swipe[j]                newCore, Pnew = updateCoreL(tt,mpt0,X,y,P0inv,d,σ) # case d=1: JLR2, case d!=1 RdJRd1                tt.cores[d] = reshape(newCore,(coreDims[d]))                P[d] = Pnew            end              end        #take dimension L out of first core        L = size(y,2)        return tt, P    end

[Monday 16:19] Eva Memmel
    function rightSuperCoreL(tt::MPT, X::Matrix, d::Int64)        D = length(tt.cores)        Gright = reshape(tt.cores[D],size(tt.cores[D])[1:2])*X' # RD x I * I  N -> RD x N         for i = D:-1:d+2            Gright = dotkron(collect(Gright'),X) #N x JRi            Gright = reshape(tt.cores[i-1],(size(tt.cores[i-1])[1],prod(size(tt.cores[i-1])[2:3])))*Gright' #Ri-1 x JRi * JRi x N -> Ri-1 x N        end        if d ==1            return Gright #R2 x N        end        Gright = dotkron(collect(Gright'),X) #N x JR3        return collect(Gright')    end

[Monday 16:20] Eva Memmel
    function leftSuperCoreL(tt::MPT, X::Matrix, d::Int64)        D = order(tt)        r1,L,r2 = size(tt.cores[1])        Gleft = reshape(tt.cores[1],L,r2)        N = size(X,1)        if d == 2            return Gleft # L x r2        end        r2,J,r3 = size(tt.cores[2])        Gleft = Gleft * reshape(tt.cores[2],r2,J*r3) #L x Ir3        Gleft = reshape(permutedims(reshape(Gleft,L,J,r3),[2,1,3]),J,L*r3) #I x Lr3        Gleft = X*Gleft # N x LR3        if d == 3            return Gleft # N x LR3        end        for i = 2:d-2            Ri,J,Ri1 = size(tt.cores[i])            Gleft = dotkron(Gleft,X) # N x JLRi1            Gleft = reshape(permutedims(reshape(Gleft,(N,J,L,Ri1)),(1, 3, 2, 4)),(N*L,J*Ri1)) # N x JLRi1 ->  NL x JRi1            Ri1,J,Ri2 = size(tt.cores[i+1])            temp = reshape(permutedims(tt.cores[i+1],(2,1,3)),(J*Ri1,Ri2)) # Ri1 x J x Ri2 -> JRi1 x Ri2            Gleft = Gleft*temp #N x LRi2            Gleft = reshape(Gleft,(N,L*Ri2)) #NL x Ri2 -> N x LRi2        end        if d == D            return dotkron(Gleft,X) # N x JLRd        end        return Gleft # N x LRd    end

[Monday 16:20] Eva Memmel
    function getUL(tt::MPT, X::Matrix, d::Int64, L::Int64)        D = length(tt.cores)        N = size(X)[1]            R1,L,R2 = size(tt.cores[1])        Rd,J,Rd1 = size(tt.cores[d])        if d == 1            Gright = rightSuperCoreL(tt, X, d) # R2 x N            Gleft = Matrix(1.0I,L,L) #L x L             superCore = kron(Gright,Gleft) # R2 x N kron L x L -> LR2 x L N            superCore = reshape(permutedims(reshape(superCore,L,R2,L,N),[4 3 1 2]),N*L,L*R2) # NL x LR2            return superCore  # NL x LR2         elseif d == 2               Gleft = leftSuperCoreL(tt,X,d) # L x R2            Gright = rightSuperCoreL(tt,X,d) # J R3 x N            superCore = kron(Gright,Gleft') # JR3 x N kron R2 x L -> R2JR3 x LN            superCore = reshape(permutedims(reshape(superCore,Rd,J,Rd1,L,N),[5 4 1 2 3]),N*L,Rd*J*Rd1) # NL x R2JR3            return superCore        elseif d == D             Gleft = leftSuperCoreL(tt,X,d) # N x JLRd                    return reshape(permutedims(reshape(Gleft,(N,J,L,Rd)),(1,3,2,4)),(N*L,J*Rd)) #NL x J*Rd        else                Gright = rightSuperCoreL(tt, X, d) # Rd1 x N            Gleft = leftSuperCoreL(tt,X,d) # N x JLRd        end        superCore = dotkron(collect(Gright'),Gleft) # N x L Rd J Rd1         superCore = reshape(reshape(superCore,N,L,Rd,J,Rd1),N*L,Rd*J*Rd1) # N L x Rd J Rd1        return superCore #N L x Rd J Rd1    end

[Monday 16:20] Eva Memmel
    function updateCoreL(tt::MPT, mpt0::MPT, X::Matrix, y::Array, P0inv::Vector{Array{Float64}}, d::Int64,σ::Float64)        #-----------------------------------------------------------        # Bayesian updating of one core while fixing all others in Volterra Setting        #        # Eva Memmel, Oktober 2021        #-----------------------------------------------------------        if isa(y,Matrix)            L = size(y,2); #L: number of outputs        else            L = 1        end        U = getUL(tt,X,d,L); #case d=1: NL x JLR2, case d!=1 Nl x RdJRd1         UTy = U'*vec(y); # case d=1: JLR2, case d!=1 RdJRd1         UTU = U'U; #case d=1: JLR2x JLR2, case d!=1 RdJRd1 RdJRd1               m0 = vec(mpt0.cores[d])  # case d=1: JLR2, case d!=1 RdJRd1        Pnew = P0inv[d] + UTU/(σ^2) #case d=1: JLR2x JLR2, case d!=1 RdJRd1 RdJRd1        mnew = pinv(Pnew)*(UTy/(σ^2) + P0inv[d]*m0)          return mnew, Pnew    end

