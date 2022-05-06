from PIL import Image
from scipy import linalg
import numpy as np
import torch
import matplotlib as plt


# Tensor Train Decomposition
def tt_decomposition(img, epsilon=0):
    # 0: load the image and convert image to correct numpy array
    img = Image.open(img)
    A = np.asarray(img)
    A = np.reshape(A, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
    # 1: compute truncation parameter delta
    d = len(A.shape)
    delta = (epsilon * linalg.norm(A)) / np.sqrt(d - 1)
    # 2: temporary tensor: C=A, r_0=1
    C = A
    r = np.zeros(d-1)
    r[0] = 1
    # 3: iterate
    tt_cores = np.zeros(d+1)
    n = A.shape
    total_error = 0
    for k in range(1, d-1):
        # 4: reshape(C, [r_(k-1)*n_k, (numel(C)/r_(k-1)*n_k)])
        print(f'r = {r}, n = {n}')
        x = int(r[k] * n[k])  # r_(k-1)*n_k
        y = int((torch.numel(torch.from_numpy(C)) / x))  # numel(C)/r_(k-1)*n_k
        C = np.reshape(C, [x, y])
        # 5: Compute delta-truncated SVD: C=USV+E, |E|_F <= delta, r_k=rank_delta(C)
        u, s, v = linalg.svd(C, full_matrices=False)
        rk = 1
        s = np.diag(s)
        error = linalg.norm((s[rk+1:]))
        if epsilon != 0:
            while error > delta:
                rk += 1
                error = linalg.norm(s[rk+1:])
                print('error', error, rk)
            total_error += error ** 2
            print(f'r{k} = {rk}')
            r[k+1] = rk
        else:
            r[k+1] = u.shape[1]
        # 6: New core: G_k:=reshape(U, [r_(k-1), n_k, r_k])
        tt_cores[k] = np.reshape(u[:, :r[k+1]], [r[k], n[k], r[k+1]])
        print(f'indices = {r[k+1]}')
        # 7: C=SV^T
        C = (s[:r[k+1]]).dot((v[:r[k+1], :]))
    tt_cores.append(np.reshape(C, [r[-1], n[-1], 1]))  # C, [r[-2], n[-1], n[-1], 1]))

    print("\n"
          "Tensor train created with order    = {d}, \n"
          "                  row_dims = {m}, \n"
          "                  col_dims = {n}, \n"
          "                  ranks    = {r}  \n"
          "                  total error   = {t}".format(d=d, m=n, n=n, r=r, t=total_error))
    return cores, A

# Tensor Train Reconstruction 2
def tt_reconstruction_2(cores,x):
    x = np.reshape(x, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
    order = len(x.shape) // 2
    row_dims = x.shape[:order]
    col_dims = x.shape[order:]
    ranks = [1] * (order + 1)

    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("The first and last rank have to be 1!")

        # reshape first core
    full_tensor = cores[0].reshape(row_dims[0] * col_dims[0], ranks[1])

    for i in range(1, order):
        # contract full_tensor with next TT core and reshape
        full_tensor = full_tensor.dot(cores[i].reshape(ranks[i], row_dims[i] * col_dims[i] * ranks[i + 1]))
        full_tensor = full_tensor.reshape(np.prod(row_dims[:i + 1]) * np.prod(col_dims[:i + 1]), ranks[i + 1])

    # reshape and transpose full_tensor
    p = [None] * 2 * order
    p[::2] = row_dims
    p[1::2] = col_dims
    q = [2 * i for i in range(order)] + [1 + 2 * i for i in range(order)]
    full_tensor = full_tensor.reshape(p).transpose(q)

    return full_tensor

def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image1,interpolation='nearest')
    plt.axis('off')
    f.add_subplot(1,2, 2)
    plt.imshow(image2,interpolation='nearest')
    plt.axis('off')
    plt.show(block=True)


dog_tensor = tt_decomposition('dog.jpg')
cores, x = dog_tensor
reconstructed_dog = tt_reconstruction_2(cores, x)
reshaped_dog = np.reshape(reconstructed_dog, (512, 512, 3))
new_image = Image.fromarray((reshaped_dog).astype(np.uint8))
old_image = Image.open('dog.jpg')
compare(old_image, new_image)