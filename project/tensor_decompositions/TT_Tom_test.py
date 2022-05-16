from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty
import matplotlib.pyplot as plt


def main():
    img = Image.open('dog.jpg_small.jpeg')
    p = np.asarray(img)
    a = p
    d = 3
    r0 = 1
    r3 = r0
    epsilon = 0.2
    delta = (epsilon / np.sqrt(d - 1)) * np.linalg.norm(a)
    n = a.shape
    r = np.zeros(d + 1)
    r[0] = r0
    r[d] = r3
    g = []
    c = a

    for k in range(d - 1):
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
    g1 = np.transpose(g[0], [1, 2, 0])
    g1 = np.squeeze(g1)
    g2 = g[1]
    g3 = np.transpose(g[d - 1])
    g3 = np.squeeze(g3)
    I = ty.tenalg.mode_dot(g2, g3, 2)
    print(I)
    I = rearrange(I)
    print(I)
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


def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis('off')
    f.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.show(block=True)


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
    return np.array(fin)


def rearrange_2(arr):
    res = np.ndarray.tolist(arr)
    res2 = [item for sublist in res for item in sublist]
    res3 = [item for sublist in res2 for item in sublist]
    fin = [[[] for _ in range(8)] for _ in range(8)]
    index_1 = 0
    index_2 = int(len(res3) // 3)
    index_3 = int(2 * (len(res3) // 3))
    for i in range(8):
        for j in range(8):
            fin[i][j].append(res3[index_1])
            fin[i][j].append(res3[index_2])
            fin[i][j].append(res3[index_3])
            index_1 += 1
            index_2 += 1
            index_3 += 1
    return np.array(fin)


if __name__ == '__main__':
    main()
