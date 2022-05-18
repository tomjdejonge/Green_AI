from T_traintex import tensortrain, tt_reconstruction
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty

img = Image.open('dog.jpg')
img2 = Image.open('baboon.png')
core, d, r, n = tensortrain(img)
B = tt_reconstruction(core, d, r, n)
# print(B.shape)

A = np.array([[1,3,5],[2,4,6]])
Y = np.array([[[1,4,7,10],[2,5,8,11], [3,6,9,12]],
                [[13,16,19,22] ,[14,17,20,23],[15,18,21,24]]])
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
    return res

# print(unfold(Y,1))
# print(unfold(Y,2))
# print(unfold(Y,3))
# print(unfold(Y,1).shape)
# print(unfold(Y,2).shape)
# print(unfold(Y,3).shape)
print(Y)
print(A)
X = A.dot(unfold(Y,1))
print(np.reshape(X,(2,2,4)))
# #nmodeproduct
# print(unfold(Y,1).dot(A.transpose(1,0)))
# print(ty.tenalg.mode_dot(Y,A,1))
# print(ty.tenalg.mode_dot(Y,A,1))


# print(np.matmul(unfold(Y,1),(A)))
# print(Y.shape, A.shape)
# print(unfold(Y,1).shape)
#
# print(unfold(Y,1).dot(A))
