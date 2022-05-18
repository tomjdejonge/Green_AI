from PIL import Image
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt


def main():
    bab = 'baboon.png'
    dog = 'dog.jpg'
    small_dog = 'dog.jpg_small.jpeg'
    original_image = Image.open(dog)
    tt, d, r, n = tt_decomposition(original_image)
    reconstructed_image = reconstruction(tt, d, r, n)
    # compare(original_image, reconstructed_image)


def tt_decomposition(tensor, epsilon=0.01):
    p = np.asarray(np.reshape(tensor, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3)))
    n = p.shape
    d = len(n)
    delta = (epsilon / np.sqrt(d - 1)) * np.linalg.norm(p)
    r = np.zeros(d + 1)
    r[0] = 1
    r[-1] = 1
    g = []
    c = p.copy()
    for k in range(d - 1):
        # print(k)
        m = int(r[k] * n[k])  # r_(k-1)*n_k
        b = int(c.size / m)  # numel(C)/r_(k-1)*n_k
        c = np.reshape(c, [m, b])
        [U, S, V] = linalg.svd(c, full_matrices=False)
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
        g.append(np.reshape(U[:, :int(r[k + 1])], [int(r[k]), int(n[k]), int(r[k + 1])]))
        p_1 = S[:int(r[k + 1]), :int(r[k + 1])]
        p_2 = V[:, :int(r[k + 1])]
        c = p_1 @ p_2.transpose()
    g.append(np.reshape(c, (int(r[d - 1]), int(n[d - 1]), int(r[d]))))
    # for i in range(len(g)):
    #     print(f'norm of core {i+1} = {linalg.norm(g[i])}')
    # print(f'norm of tensor = {linalg.norm(p)}')
    return g, d, r, n


def reconstruction(g, d, r, n):
    r = list(r.astype(np.uint))
    # for i in range(len(r)):
    #     r[i] = int(r[i])
    # print(f'r= {r}')
    n = list(n)
    full_tensor = np.reshape(g[0], (int(n[0]), int(r[1])))

    for k in range(1, d):
        full_tensor = full_tensor.dot(g[k].reshape(int(r[k]), int(n[k]) * int(r[k + 1])))
        full_tensor = full_tensor.reshape(np.prod(n[:k + 1]), int(r[k + 1]))
    # print(full_tensor.shape)
    # q = [2 * i for i in range(d // 2)] + [1 + 2 * i for i in range(d // 2)]
    # print(f'q = {q}')
    # full_tensor = full_tensor.reshape(n).transpose(q)
    # full_tensor = full_tensor * 255 / (abs(full_tensor.min()) + full_tensor.max())
    z = np.array(np.reshape(full_tensor, (512, 512, 3)))
    print(z.astype(np.uint8))
    return Image.fromarray(z.astype(np.uint8), 'RGB')

def reconstruction_2(g, d, r, n):
    a = r[0]
    T = g[0]
    for i in range(d-1):
        x = T.shape
        e = np.reshape(T, (a*n[i], r[i+1]))


def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis('off')
    f.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.show(block=True)


if __name__ == '__main__':
    main()
