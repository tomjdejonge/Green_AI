
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty

def tensortrain(img, epsilon=0):
    img = np.array(Image.open('dog.jpg'))
    img = np.asarray(np.reshape(img, (4,4,4,4,4,4,4,4,4,3)))
    n = img.shape
    d = len(n)
    print(d)
    delta = (epsilon / np.sqrt(d - 1)) * np.linalg.norm(img)

    r = np.zeros(d + 1)
    r[0] = 1
    r[-1] = 1
    g = []
    c = img.copy()

    for k in range(d - 1):
        print(k)
        m = int(r[k] * n[k])  # r_(k-1)*n_k
        b = int(c.size / m)  # numel(C)/r_(k-1)*n_k
        c = np.reshape(c, [m, b], order="F")
        [U, S, V] = linalg.svd(c, full_matrices=False)
        V = V.transpose()
        S = np.diag(S)
        s = np.diagonal(S)
        s = np.reshape(s, (s.shape[0], 1))
        rank = 0
        error = np.linalg.norm(s[rank])
        while error > delta:
            rank += 1
            error = np.linalg.norm(s[rank + 1:])
        r[k + 1] = rank + 1
        g.append(np.reshape(U[:, :int(r[k + 1])], [int(r[k]), int(n[k]), int(r[k + 1])]))
        p_1 = S[:int(r[k + 1]), :int(r[k + 1])]
        p_2 = V[:, :int(r[k + 1])]
        c = p_1 @ p_2.transpose()
    g.append(np.reshape(c, (int(r[d - 1]), int(n[d - 1]), int(r[d])), order="F"))
    # print(g)
    return g, d

def tt_reconstruction(g, d):
    g1 = np.transpose(g[0], [1, 2, 0])
    g1 = np.squeeze(g1)
    g2 = g[1]
    g3 = np.transpose(g[d - 1])
    g3 = np.squeeze(g3)
    I = ty.tenalg.mode_dot(g2, g3, 2)

    B = []
    for i in range(0, 3):
        res = np.matmul(g1, (I[:, :, i]))
        B.append(res)
    B = np.array(B)
    B = np.squeeze(B)
    B = np.transpose(B)
    c = [[], [], []]
    for i in range(3):
        c[i] = (np.transpose(B[:, :, i]))
    c = np.array(c)
    Dog = (c.astype(np.uint8))
    print(Dog)
    Dog = rearrange_2(Dog)
    new_image = Image.fromarray(Dog.astype(np.uint8))
    old_image = img
    compare(old_image, new_image)
    return B

def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image1)
    plt.axis('off')
    f.add_subplot(1,2,2)
    plt.imshow(image2.rotate(-90))
    plt.axis('off')
    plt.show(block=True)


img = Image.open('dog.jpg')
# img2 = Image.open('baboon.png')
core, d = tensortrain(img)
B = tt_reconstruction(core, d)

new_image = Image.fromarray((B).astype(np.uint8))
old_image = img

compare(old_image, new_image)