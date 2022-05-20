import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty


def tensortrain(img, epsilon=0.1):
    img = np.asarray(img)
    print(img.shape)

    img = np.reshape(img, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
    n = img.shape
    print(n)
    d = len(n)

    delta = (epsilon / np.sqrt(d - 1)) * np.linalg.norm(img)

    r = np.zeros(d + 1)
    r[0] = 1
    r[-1] = 1
    g = []
    c = img.copy()

    for k in range(d - 1):

        m = int(r[k] * n[k])  # r_(k-1)*n_k
        b = int(c.size / m)  # numel(C)/r_(k-1)*n_k
        c = np.reshape(c, [m, b])
        [U, S, V] = linalg.svd(c, full_matrices=False)
        V = V.transpose()
        S = np.diag(S)
        s = np.diagonal(S)
        s = np.reshape(s, (s.shape[0], 1))

        rank = 0
        error = np.linalg.norm(s[rank + 1])
        while error > delta:
            rank += 1
            error = np.linalg.norm(s[rank + 1:])
        r[k + 1] = rank + 1

        g.append(np.reshape(U[:, :int(r[k + 1])], [int(r[k]), int(n[k]), int(r[k + 1])]))
        p_1 = S[:int(r[k + 1]), :int(r[k + 1])]
        p_2 = V[:, :int(r[k + 1])]
        c = p_1 @ p_2.transpose()

    g.append(np.reshape(c, (int(r[- 2]), int(n[- 1]), int(r[-1]), 1)))

    # for i in range(len(g)):
    #     print(f'norm of core {i+1} = {linalg.norm(g[i])}')
    # print(f'norm of core = {linalg.norm(img)}')
    return g, d, r, n


def tt_reconstruction(cores, d, r, n):
    r = list(r.astype(np.uint))

    n = list(n)
    full_tensor = np.reshape(cores[0], (int(n[0]), int(r[1])))

    for k in range(1, d):
        full_tensor = full_tensor.dot(cores[k].reshape(int(r[k]), int(n[k]) * int(r[k + 1])))
        full_tensor = full_tensor.reshape(np.prod(n[:k + 1]), int(r[k + 1]))

    # q = [2* i for i in range(d//2)] + [1+2*i for i in range(d//2)]

    return np.array(np.reshape(full_tensor, (512, 512, 3)))


def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis('off')
    f.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.show()


def border(img, low, high, p=False):
    lowcount = 0
    highcount = 0

    for x in range(len(img)):
        for y in range(len(img[0])):
            for z in range(len(img[0][0])):
                if img[x][y][z] <= low:
                    lowcount += 1
                    img[x][y][z] = low - img[x][y][z]

                elif img[x][y][z] >= high:
                    highcount += 1
                    # print(B[x][y][z])
                    img[x][y][z] = 2* high - img[x][y][z]

    print(lowcount,highcount)

    return img


img2 = Image.open('baboon.png')
img = Image.open('dog.jpg')

core, d, r, n = tensortrain(img)

B = tt_reconstruction(core, d, r, n)
B = border(np.array(B), 0, 255, p=True)

new_image = Image.fromarray((B).astype(np.uint8), 'RGB')
old_image = img

compare(old_image, new_image)
