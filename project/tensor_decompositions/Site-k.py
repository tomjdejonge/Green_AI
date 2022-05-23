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
# for i in range(len(core)):
#     print(core[i].shape)

v = linalg.norm(core[d-1])
K = 9
C = len(core)-1
# for i in range(len(core)):
#     print(i,'bbb', np.array(core[i]).shape)

def sitek(core):
    v = linalg.norm(core[d - 1])
    C = len(core) - 1
    K = core[-1]
    for i in range(1,C-1):
        # print(i)
        C = core[i]
        [Left, Foot, Right] = C.shape

        C = unfold(C,1)
        # print(C.shape)

        [Q,R] = np.linalg.qr(C, mode='reduced')

        print(R.shape[0], Foot, int(Q.size / (Foot * R.shape[0])))
        # print(R.shape)
        core[i] = np.reshape(Q,(R.shape[0], Foot, int(Q.size/(Foot*R.shape[0]))))
        # print(core[i].shape)
        core[i-1] = mode_dot(core[i-1],R,3)
        K = i-1
        return core

def sitek2(core):
    K = core[-1]
    d = len(core)-1
    for i in range(1, d-1):
        # print(i)
        C = core[i]
        # print(C.shape)
        [Left, Foot, Right] = C.shape

        [Q, R] = np.linalg.qr(C, mode='reduced')
        # print(core[i-1])
        print(i, len(R.transpose()))

        core[i] = np.reshape(Q, (R.shape[0], Foot, int(Q.size / (Foot * R.shape[0]))))
        # core[i-1] = mode_dot(core[i-1],np.array(R),3)
    return core

print(len(sitek2(core)))

# print(sitek(core))
