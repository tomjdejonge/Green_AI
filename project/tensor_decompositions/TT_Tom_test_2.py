from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty
import matplotlib.pyplot as plt


img = Image.open('dog.jpg_small.jpeg')
p = np.asarray(img)
# print(p.shape)
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
    # print('c =', c)
    # print(c.shape)
    [U, S, V] = linalg.svd(c, full_matrices=False)
    V = V.transpose()
    S = np.diag(S)
    s = np.diagonal(S)
    s = np.reshape(s, (s.shape[0], 1))
    # print(s.shape)
    # print(s)
    # print('s = ', s)
    # print('U = ', U)
    # print('S = ', S)
    # print('V = ', V)

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
    x = np.reshape(U[:, :int(r[k+1])], [int(r[k]), int(n[k]), int(r[k+1])], order="F")
    # print(x)
    # print(x.shape)
    # print(np.linalg.norm(g[k]))
    # print(S.shape)
    p_1 = S[:int(r[k + 1]), :int(r[k + 1])]
    # print(p_1)
    p_2 = V[:, :int(r[k + 1])]
    # print(p_2)
    c = p_1 @ p_2.transpose()
    for i in range(len(g)):
        print(f'norm of core {i + 1} = {linalg.norm(g[i])}')
    # print(c)
# print(r[d-1])
# print(n[d-1])
# print(r[d])
g.append(np.reshape(c, (int(r[d-1]), int(n[d-1]), int(r[d])), order="F"))
# print(g[1])
# print(g[2])
g1 = np.transpose(g[0], [1, 2, 0])
g1 = np.squeeze(g1)
# print(g1)
# print(g1.shape)
g2 = g[1]
# print(g2)
# print(g2.shape)
g3 = np.transpose(g[d-1])
g3 = np.squeeze(g3)
# print(g3)
# print(g3.shape)

# print(g3)
# print(g3.shape)


def rearrange(arr):
    res = np.ndarray.tolist(arr)
    res2 = [item for sublist in res for item in sublist]
    fin = [[], [], []]
    for i in range(len(res[0])):
        i1 = res2[0]
        i2 = res2[1]
        i3 = res2[2]
        fin[0].append(i1)
        fin[1].append(i2)
        fin[2].append(i3)
        res2 = res2[3:]
    return (np.array(fin))


I = ty.tenalg.mode_dot(g2, g3, 2)
for i in range(3):
    I[:, :, i] = rearrange(I[:, :, i])


# print(I.shape)

B = []
for i in range(0, 3):
    res = np.matmul(g1, (I[:, :, i]))
    B.append(res)

B = np.array(B)
B = np.squeeze(B)
B = np.transpose(B)

# for i in range(3):
#     print(np.transpose(B[:, :, i]), '\n')

c = [[],[],[]]
for i in range(3):
    c[i] = (np.transpose(B[:, :, i]))
c = np.array(c)

# print(c)

Dog = (c.astype(np.uint8))
print(Dog)

new_image = Image.fromarray(Dog)
old_image = img
print(f'norm of image is {linalg.norm(img)}')

def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image1)
    plt.axis('off')
    f.add_subplot(1,2,2)
    plt.imshow(image2)
    plt.axis('off')
    plt.show(block=True)

compare(old_image, new_image)