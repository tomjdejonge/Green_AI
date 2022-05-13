from T_traintex import tensortrain, tt_reconstruction
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty

img = Image.open('dog.jpg')
img2 = Image.open('baboon.png')
core, d = tensortrain(img)
B = tt_reconstruction(core, d)

def unfold(tensor, n):
    print(tensor[0].shape, tensor[1].shape, tensor[2].shape)
    tensor = np.array([tensor[0], tensor[1], tensor[2]])
    x = tensor.shape
    x =
    print('aa' ,tensor.shape)
    if n ==2:
        tensor = np.pagetranspose(np.transpose(tensor,2,1,0))
    elif n == 3:
        tensor = np.transpose(tensor, [2,0,1])   #[tensor[2], tensor[0], tensor[1]]
    return np.reshape(tensor,x[n],[])

print(unfold(core, d))