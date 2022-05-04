from PIL import Image
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import torch


# Tensor Train Decomposition
def tt_decomposition(img, epsilon=0.5):
    # Load the image and convert image to correct numpy array
    img = Image.open(img)
    x = np.asarray(img)
    print(f'The shape of the original image is: {x.shape}')
    x = np.reshape(x, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
    print('The shape of the initially reshaped image is:', x.shape)
    # Define order, row_dims, col_dims, ranks and cores
    order = len(x.shape) // 2
    row_dims = x.shape[:order]
    col_dims = x.shape[order:]
    ranks = [1] * (order + 1)
    cores = []
    terror = 0
    delta = epsilon / (np.sqrt(order*2)) * linalg.norm(x)
    print(f'delta = {delta}')
    # permute dimensions, e.g., for order = 4: p = [0, 4, 1, 5, 2, 6, 3, 7]
    p = [order * j + i for i in range(order) for j in range(2)]

    print(f'p= {p}, zelf = {p[:3]}')
    y = np.transpose(x, p).copy()

    # decompose the full tensor
    for i in range(order - 1):
        # reshape residual tensor
        m = ranks[i] * row_dims[i] * col_dims[i]            #r_(k-1)*n_k
        n = int((torch.numel(torch.from_numpy(y))  /m))     # numel(C)/r_(k-1)*n_k    or  n = row_dims[i + 1:]) * np.prod(col_dims[i + 1:]
        y = np.reshape(y, [m, n])

        # apply SVD in order to isolate modes
        [u, s, v] = svd(y)                                  #svd
        rk = 1
        s = np.diag(s)
        error = linalg.norm((s[rk+1:]))
        print(f'error {i} = {error}')
        while error > delta:
            rk += 1
            error = linalg.norm(s[rk+1:])

        terror = terror + error**2
        print(f'rk = {rk}')
        # define new TT core
        ranks[i + 1] = u.shape[1]
        cores.append(np.reshape(u, [ranks[i], row_dims[i], col_dims[i], ranks[i + 1]]))

        # set new residual tensor
        # y = s.dot(v)            # S * V^T
        y = (s[1:rk,1:rk]).dot(v[:,1:rk])

    # define last TT core
    cores.append(np.reshape(y, [ranks[-2], row_dims[-1], col_dims[-1], 1]))
    print(len(cores))
    print("\n"
          "Tensor train created with order    = {d}, \n"
          "                  row_dims = {m}, \n"
          "                  col_dims = {n}, \n"
          "                  ranks    = {r}  \n"
          "                  terror   = {t}"    .format(d=order, m=row_dims, n=col_dims, r=ranks, t=terror))
    return cores, row_dims, col_dims, ranks, order


# Single Value Decomposition
def svd(x):
    u, s, v = linalg.svd(x, full_matrices=False)
    return u, s, v


dog_tensor = tt_decomposition('dog.jpg')
cores, row_dims, col_dims, ranks, order = dog_tensor


# Tensor Train Reconstruction 1
def tt_reconstruction_1(cores, row_dims, col_dims, ranks, order):
    tt_mat = cores[0].reshape(row_dims[0], col_dims[0], ranks[1])

    for i in range(1, order):
        # contract tt_mat with next TT core, permute and reshape
        tt_mat = np.tensordot(tt_mat, cores[i], axes=(2, 0))
        tt_mat = tt_mat.transpose([0, 2, 1, 3, 4]).reshape((np.prod(row_dims[:i + 1]),
                                                            np.prod(col_dims[:i + 1]), ranks[i + 1]))

    # reshape into vector or matrix
    m = np.prod(row_dims)
    n = np.prod(col_dims)
    if n == 1:
        tt_mat = tt_mat.reshape(m)
    else:
        tt_mat = tt_mat.reshape(m, n)

    return tt_mat

# Tensor Train Reconstruction 2
def tt_reconstruction_2(cores, row_dims, col_dims, ranks, order):
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("The first and last rank have to be 1!")

        # reshape first core
    full_tensor = cores[0].reshape(row_dims[0] * col_dims[0], ranks[1])

    for i in range(1, order):
        # contract full_tensor with next TT core and reshape
        full_tensor = full_tensor.dot(cores[i].reshape(ranks[i],
                                                            row_dims[i] * col_dims[i] * ranks[i + 1]))
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

def are_images_equal(im1, im2):
    if list(im1.getdata()) == list(im2.getdata()):
        print("\nThe images are identical")
    else:
        print("\nThe images are different")

def pixelcount(img):
    count = 0
    for y in range(img.height):
        for x in range(img.width):
            count += 1

    return count

reconstructed_dog = tt_reconstruction_2(cores, row_dims, col_dims, ranks, order)
print(f'\nThe shape of the reconstructed tensor is: {reconstructed_dog.shape}')
reshaped_dog = np.reshape(reconstructed_dog, (512, 512, 3))
new_image = Image.fromarray((reshaped_dog).astype(np.uint8))
old_image = Image.open('dog.jpg')

# print(f'Pixels in new image {pixelcount(new_image)}')
# print(f'Pixels in old image {pixelcount(old_image)}')

# are_images_equal(new_image, old_image)

# compare(old_image,new_image)
print(f'\nThe shape of the reconstructed tensor is: {reconstructed_dog.shape}')
reshaped_dog = np.reshape(reconstructed_dog, (512, 512, 3))
new_image = Image.fromarray((reshaped_dog).astype(np.uint8))
old_image = Image.open('dog.jpg')

print(f'Pixels in new image {pixelcount(new_image)}')
print(f'Pixels in old image {pixelcount(old_image)}')

are_images_equal(new_image, old_image)

compare(old_image,new_image)
