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


def unfold(tensor, x):

    if x==1:
        shape = tensor.shape
        res = np.reshape(tensor, (shape[0],shape[1]*shape[2]))
    elif x==2:
        tensor = tensor.transpose(1,0,2)
        shape = tensor.shape
        res = np.reshape(tensor, (shape[0],shape[1]*shape[2]))
    elif x==3:
        tensor = tensor.transpose(2, 0, 1)
        shape = tensor.shape
        res = np.reshape(tensor, (shape[0], shape[1] * shape[2]))
    return res

#nmodeproduct
A = np.array([[1,2,3],[4,5,6]])
Y = np.array([[[1,4,7,10], [13,16,19,22]],
 [[2,5,8,11], [14,17,20,23]],
 [[3,6,9,12], [15,18,21,24]]])

print(unfold(Y,1).dot(A))

