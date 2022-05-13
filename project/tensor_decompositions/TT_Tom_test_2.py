from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty
# import matplotlib.pyplot as plt


img = Image.open('dog.jpg')
p = np.asarray(img)
a = p

d = 3
r0 = 1
r3 = r0
epsilon = 0.2
delta = (epsilon / np.sqrt(d-1)) * np.linalg.norm(a)

n = a.shape
r = np.zeros(d+1)
r[0] = r0
r[d] = r3
g = []

c = a

for k in range(d-1):
    m = int(r[k] * n[k])  # r_(k-1)*n_k
    b = int(c.size / m)  # numel(C)/r_(k-1)*n_k
    # print(m)
    # print(b)
    c = np.reshape(c, [m, b], order="F")
    # print('c = ', c[-5:])
    [U, S, V] = linalg.svd(c, full_matrices=False)
    V = V.transpose()
    S = np.diag(S)
    s = np.diagonal(S)
    s = np.reshape(s, (s.shape[0], 1))
    # print(s.shape)
    # print(s)
    # print('s = ', s.shape)
    # print('U = ', U.shape)
    # print('S = ', S.shape)
    # print('V = ', V.shape)

    rank = 0
    # print(S[rank])
    error = np.linalg.norm(s[rank])
    # print(error)
    while error > delta:
        rank += 1
        error = np.linalg.norm(s[rank+1:])
    r[k+1] = rank + 1
    # print(int(r[k+1]))
    # print(int(n[k]))
    # print(int(r[k]))
    # print((U[:, :int(r[k+1])]).shape)
    g.append(np.reshape(U[:, :int(r[k+1])], [int(r[k]), int(n[k]), int(r[k+1])]))
    # print(np.linalg.norm(g[k]))
    # print(S.shape)
    p_1 = S[:int(r[k + 1]), :int(r[k + 1])]
    # print(p_1.shape)
    p_2 = V[:, :int(r[k + 1])]
    # print(p_2.shape)
    c = p_1 @ p_2.transpose()
    # print(c)
# print(r[d-1])
# print(n[d-1])
# print(r[d])
g.append(np.reshape(c, (int(r[d-1]), int(n[d-1]), int(r[d]))))
# print(g[0].shape)
# print(g[1].shape)
# print(g[2].shape)
g1 = np.transpose(g[0], [1, 2, 0])
g1 = np.squeeze(g1)
g2 = g[1]
g3 = np.transpose(g[d-1])
g3 = np.squeeze(g3)

# print(g1.shape, g2.shape, g3.shape)

I = ty.tenalg.mode_dot(g2, g3, 2)
print(I.shape)

B = []
for i in range(0,3):

    res = np.matmul((I[:,:,i]),g1)
    B.append(res)

B = np.array(B)
B = np.squeeze(B)
B = np.transpose(B)

print(B.shape)
