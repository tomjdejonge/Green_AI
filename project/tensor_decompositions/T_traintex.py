
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty

def tensortrain(img, d = 3, epsilon = 0.2):
    a = np.asarray(img)

    delta = (epsilon / np.sqrt(d-1)) * np.linalg.norm(a)

    n = a.shape
    r = np.zeros(d+1)
    r[0] = 1
    r[d] = 1
    g = []

    c = a

    for k in range(d-1):
        m = int(r[k] * n[k])  # r_(k-1)*n_k
        b = int(c.size / m)  # numel(C)/r_(k-1)*n_k
        c = np.reshape(c, [m, b])

        [U, S, V] = linalg.svd(c, full_matrices=False)
        V = V.transpose()
        s = np.diag(S)
        rank = 0

        error = np.linalg.norm(s[rank])

        while error > delta:
            rank += 1
            error = np.linalg.norm(s[rank+1:])
        r[k+1] = rank + 1

        g.append(np.reshape(U[:, :int(r[k+1])], [int(r[k]), int(n[k]), int(r[k+1])]))

        p_1 = s[:int(r[k + 1]), :int(r[k + 1])]
        p_2 = V[:, :int(r[k + 1])]

        c = p_1 @ p_2.transpose()

    g.append(np.reshape(c, (int(r[d-1]), int(n[d-1]), int(r[d]))))
    return g, d

def tt_reconstruction(g, d):

    g1 = np.transpose(g[0], [1, 2, 0])

    g2 = g[1]

    g3 = np.transpose(g[d-1])
    g3 = np.squeeze(g3)


    x = ty.tenalg.mode_dot(g2, g3, 2)



    B = []
    for i in range(0,3):

        res = np.matmul(np.transpose(x[:,:,i]),g1)
        B.append(res)

    B = np.array(B)
    B = np.squeeze(B)
    B = np.transpose(B)

    return B




img = Image.open('dog.jpg')
core, d = tensortrain(img)
B = tt_reconstruction(core, d)

new_image = Image.fromarray((B).astype(np.uint8))
old_image = img

def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image1,interpolation='nearest')
    plt.axis('off')
    f.add_subplot(1,2, 2)
    plt.imshow(image2,interpolation='nearest')
    plt.axis('off')
    plt.show(block=True)

compare(old_image, new_image)