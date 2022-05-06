from PIL import Image
from scipy import linalg
import numpy as np
import torch
# import matplotlib.pyplot as plt


# Tensor Train Decomposition
def tt_decomposition(img, epsilon=0):
    # load the image and convert image to correct numpy array
    img = Image.open(img)
    a = np.asarray(img)
    a = np.reshape(a, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
    # define parameters
    d = len(a.shape)
    tt_cores = np.zeros(d + 1)
    n = a.shape
    # 1: compute truncation parameter delta
    delta = (epsilon * linalg.norm(a)) / np.sqrt(d - 1)
    # 2: temporary tensor: c=a, r_0=1
    c = a
    r = np.zeros(d+1)
    r[0] = 1
    total_error = 0
    # 3: iterate
    for k in range(d-1):
        #print(f'r = {r}, n = {n}')
        x = int(r[k] * n[k])  # r_(k-1)*n_k
        y = int((torch.numel(torch.from_numpy(c)) / x))  # Numel(c)/r_(k-1)*n_k
        c = np.reshape(c, [x, y])
        # 4: SVD
        u, s, v = linalg.svd(c, full_matrices=False)
        rk = 1
        singular_values = np.diag(s)
        error = np.linalg.norm(singular_values[rk+1:])
        while error > delta:
            rk += 1
            error = np.linalg.norm(singular_values[rk + 1:])
        total_error += error**2
        r[k+1] = rk
        print(u[:int(r[k+1])])
        print(r[k])
        print(n[k])
        print(int(r[k+1]))
        print(u.shape)
        tt_cores[k] = np.reshape(u[:, int(r[k+1])], [int(r[k]), int(n[k]), int(r[k+1])])
        s_1 = s[:int(r[k+1]), :int(r[k+1])]
        v_transposed = (v[:, int(r[k+1])]).transpose
        c = np.matmul(s_1, v_transposed)
    k = d-1
    tt_cores[k+1] = np.reshape(c, (r[k+1], n[k+1], 1))
    tt_cores[k+2] = d
    rel_error = np.sqrt(total_error) / np.linalg.norm(a)

    print("\n"
          "Tensor train created with order    = {d}, \n"
          "                  row_dims = {m}, \n"
          "                  col_dims = {n}, \n"
          "                  ranks    = {r}  \n"
          "                  terror   = {t}".format(d=d, m=n, n=n, r=r, t=rel_error))
    return tt_cores, n, r, d


tt_decomposition('dog.jpg')
