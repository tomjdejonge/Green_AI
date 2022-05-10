from PIL import Image
from scipy import linalg
import numpy as np
import torch
import matplotlib.pyplot as plt


# Tensor Train Decomposition
def tt_decomposition(img, epsilon: float = 0):
    # load the image and convert image to correct numpy array
    img = Image.open(img)
    original = np.asarray(img)
    print(f'shape of original image: {original.shape}')
    a = np.reshape(original, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
    print(f'shape of reshaped image: {a.shape}')
    # define parameters
    n = a.shape
    d = np.ndim(a)
    tt_cores = []
    # 1: compute truncation parameter delta
    delta = (epsilon / np.sqrt(d - 1)) * linalg.norm(original)
    print(f'd: {d}\nn: {n}\ndelta: {delta}')
    # 2: temporary tensor: c=a, r_0=1
    c = a
    r = [0]*d
    r[0] = 1
    print(f'r: {r}')
    total_error = 0
    # 3: iterate
    for k in range(d-1):
        print(f'r = {r}')
        x = int(r[k] * n[k])  # r_(k-1)*n_k
        y = int((torch.numel(torch.from_numpy(c))/x))  # Numel(c)/r_(k-1)*n_k
        c = np.reshape(c, [x, y])
        # 4: SVD
        u, s, v = linalg.svd(c, full_matrices=False)
        rk = 1
        s = np.diag(s)
        print(f's_min: {s.min()}\ns_max: {s.max()}')
        error = linalg.norm((s[rk+1:]))
        print(f'error before iteration: {error}')
        if epsilon != 0:
            while error > delta:
                print(f'rk: {rk}\nerror: {error}')
                rk += 1
                error = linalg.norm((s[rk+1:]))
            print(f'error after iteration: {error}')
            print(f'rk: {rk}')
            r[k+1] = rk
        else:
            r[k+1] = u.shape[1]
        total_error += error**2
        # print(u.shape)
        # print(r[k])
        # print(n[k])
        # print(r[k+1])
        tt_cores.append(np.reshape(u, [r[k], n[k], r[k+1]]))

        # s_1 = s[:, :r[k+1]]  # , :(r[k+1])]
        # v_transposed = (v[:r[k+1]]).transpose
        # c = np.matmul(s_1, v_transposed)
        c = np.matmul((s[:r[k+1]]), (v[:r[k+1], :]))
    tt_cores.append(np.reshape(c, (r[-1], n[-1], 1)))
    rel_error = np.sqrt(total_error) / np.linalg.norm(a)

    print("\n"
          "Tensor train created with order    = {d}, \n"
          "                  row_dims = {m}, \n"
          "                  col_dims = {n}, \n"
          "                  ranks    = {r}  \n"
          "                  delta    = {delta}  \n"
          "                  rel_error   = {t}".format(d=d, m=n, n=n, r=r, t=rel_error, delta=delta))
    return tt_cores, n, r, d


# Tensor Train Reconstruction 2
def tt_reconstruction(cores, n, r, d):

    # reshape first core
    full_tensor = cores[0].reshape(n[0], r[1])
    # print(f'd={d}')
    for i in range(1, d-1):
        # contract full_tensor with next TT core and reshape
        full_tensor = full_tensor.dot(cores[i].reshape(r[i], n[i] * r[i+1]))
        full_tensor = full_tensor.reshape(np.prod(n[:i + 1]), r[i+1])

    # reshape and transpose full_tensor
    p = [None] * 2 * d
    p[::2] = n
    p[1::2] = n
    q = [2 * i for i in range(d//2)] + [1 + 2 * i for i in range(d//2)]
    # print(q)
    full_tensor = full_tensor.reshape(n).transpose(q)

    return full_tensor


def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(image1, interpolation='nearest')
    plt.axis('off')
    f.add_subplot(1, 2, 2)
    plt.imshow(image2, interpolation='nearest')
    plt.axis('off')
    plt.show(block=True)


dog_tt_cores, dog_n, dog_r, dog_d = tt_decomposition('dog.jpg', 0.5)
reconstructed_dog = tt_reconstruction(dog_tt_cores, dog_n, dog_r, dog_d)
reshaped_dog = np.reshape(reconstructed_dog, (512, 512, 3))
old_image = Image.open('dog.jpg')
new_image = Image.fromarray(reshaped_dog.astype(np.uint8))
compare(old_image, new_image)
