from T_traintex import tensortrain, tt_reconstruction
from N_mode import unfold, nmultiplication
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
def sitek(core):
    v = linalg.norm(core[d - 1])
    K = 9
    C = len(core) - 1
    for i in range(k):
        C = core[i]
        [Left, Foot, Right] = C.size
        C = unfold(C,1)
        [Q,R] = np.linalg.qr(C, mode='reduced')
        core[i] = np.reshape(Q,(R.size, Foot, Q.size/(Foot*R.size)))
        core[i-1] = nmultiplication(core[i-1],R,3)
        return core

print(sitek(core))
