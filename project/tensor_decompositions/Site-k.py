from T_traintex import tensortrain, tt_reconstruction
from N_Mode import unfold, mode_dot
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np

import tensorly as ty

img = Image.open('dog.jpg')
img2 = Image.open('baboon.png')
core, d, r, n = tensortrain(img)
B = tt_reconstruction(core, d, r, n)

v = linalg.norm(core[d-1])
K = 9
C = len(core)-1
# for i in range(len(core)):
#     print(i,'bbb', np.array(core[i]).shape)

def sitek(core):
    v = linalg.norm(core[d - 1])
    C = len(core) - 1
    for i in range(C):
        print(i)
        C = core[i]
        [Left, Foot, Right] = C.shape
        C = unfold(C,1)
        # print(C.shape)

        [Q,R] = np.linalg.qr(C, mode='reduced')
        # print(R.shape)
        core[i] = np.reshape(Q,(R.shape[0], Right, int(Q.size/(Foot*R.shape[0]))))
        # print(core[i].shape)
        core[i-1] = mode_dot(core[i],R,3)
        K = i-1
        return core

print(sitek(core))
