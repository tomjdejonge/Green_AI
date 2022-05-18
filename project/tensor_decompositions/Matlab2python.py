from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty
import matplotlib.pyplot as plt

img = Image.open('baboon.png')
image_sequence = img.getdata()
P = np.array(image_sequence)
A = P.reshape([4, 4, 4, 4, 4, 4, 4, 4, 4, 3], order='F').copy()

# Variables
d = len(A.shape)
epsilon = 0.1
delta = (epsilon / np.sqrt(d - 1)) * np.linalg.norm(A)

# Arrays
n = A.shape
r = np.zeros(d + 1)
r[0] = 1
r[-1] = 1
tt = []

# Loop
C = A
for k in range(d-1):
    m = int(r[k] * n[k])  # r_(k-1)*n_k
    b = int(C.size / m)  # numel(C)/r_(k-1)*n_k
    C = np.reshape(C, [m, b])
    [U, S, V] = linalg.svd(C, full_matrices=False)
    V = V.transpose()
    S = np.diag(S)
    s = np.diagonal(S)
    s = np.reshape(s, (s.shape[0], 1))
    # print(c.shape)
    rank = 0
    error = np.linalg.norm(s[rank + 1])
    while error > delta:
        rank += 1
        error = np.linalg.norm(s[rank + 1:])
    r[k + 1] = rank + 1
    # print(k,r)
    tt.append(np.reshape(U[:, :int(r[k + 1])], [int(r[k]), int(n[k]), int(r[k + 1])]))
    p_1 = S[:int(r[k + 1]), :int(r[k + 1])]
    p_2 = V[:, :int(r[k + 1])]
    C = p_1 @ p_2.transpose()
tt.append(np.reshape(C, (int(r[d - 1]), int(n[d - 1]), int(r[d]))))
for i in range(len(tt)):
    print(f'norm of core {i+1} = {linalg.norm(tt[i])}')
print(f'norm of tensor = {linalg.norm(P)}')