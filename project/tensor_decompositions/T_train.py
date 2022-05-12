from PIL import Image
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import torch
 ## ucidatasets
def tt_decomposition(img, epsilon=0.01):
    # Load the image and convert image to correct numpy array
    img = Image.open(img)
    x = np.asarray(img)
    # print(x)
    print(f'The shape of the original image is: {x.shape}')
    C = x
    print('The shape of the initially reshaped image is:', C.shape)
    n = C.shape
    print(n)
    d = 3
    cores = [0] * (d)
    delta1 = epsilon / (np.sqrt(d-1)) * np.linalg.norm(x)
    delta =  epsilon / (np.sqrt(d-1)) * fro(x)
    delta2 = epsilon / (np.sqrt(d-1)) * frob(x)
    # print(f' delta = {delta}, delta1 = {delta1} , delta2 = {delta2}')
    r = [0] * (d+1)
    r[0] = 1


    for i in range(d-1):
        m = int(r[i] * n[i])                                   # r_(k-1)*n_k
        b = int(C.size/m)                                   # numel(C)/r_(k-1)*n_k    or  n = row_dims[i + 1:]) * np.prod(col_dims[i + 1:]

        C = np.reshape(C, [m, b])
        terror = 0

        [u, s, v] = linalg.svd(C, full_matrices=False)          #(u @ np.diag(s) @ v).astype(int)

        rk = 0

        error = linalg.norm((s[rk+1:]))
        print(f'{i,u.shape,s.shape,v.shape}')
        if epsilon != 0:
            while error > delta:
                rk +=1
                error = linalg.norm(s[rk+1:])
            # print(f'error = {error}')
            print(f'r{i} = {r}')
        else:
            rk = 0
        s = np.diag(s)
        r[i+1] = rk
        terror += error ** 2
        # cores.append(np.reshpa)
        cores[i] = (np.reshape(u[:,:r[i+1]], [r[i], n[i], r[i + 1]]))

        C = (s[:r[i+1],:r[i+1]])@(v[:r[i+1],:])


    cores[-1] = (np.reshape(C, [r[-1], n[-1], 1]))       #C, [r[-2], n[-1], n[-1], 1]))
    rerror = np.sqrt(terror)/np.linalg.norm(x)
    # print(f'cores = {np.linalg.norm(cores[0])}, original tensor = {np.linalg.norm(x)}')
    for i in range(d):
        print(i, np.linalg.norm(cores[i]), np.linalg.norm(x))
    print(cores[1])
    # print('coress=',cores[-7])
    print("\n"
          "Tensor train created with order    = {d}, \n"
          "                  row_dims = {m}, \n"
          "                  col_dims = {n}, \n"
          "                  ranks    = {r}  \n"
          "                  relative error   = {t}"    .format(d=d, m=n, n=n, r=r, t=rerror))
    return cores, n, r, d, C


# Single Value Decomposition
def svd(x):
    u, s, v = linalg.svd(x, full_matrices=False)
    return u, s, v

def fro(x):
    count = 0
    res = 0
    for i in range(len(x)):
        for j in range(len(x)):
            for k in range(3):
                res += pow(x[i][j][k], 2)
                count += 1
    return round(np.sqrt(res),4)

def frob(tensor):
    tensor = tensor.flatten()
    return np.sqrt(np.matmul(tensor.transpose(), tensor))

dog_tensor = tt_decomposition('dog.jpg')
cores, n, r, d, C = dog_tensor


# Tensor Train Reconstruction 2
def tt_reconstruction_2(cores, n, r, d, C):

        # reshape first core

    full_tensor = []
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

reconstructed_dog = tt_reconstruction_2(cores, n,r,d, C)

print(f'\nThe shape of the reconstructed tensor is: {reconstructed_dog.shape}')
reshaped_dog = np.reshape(reconstructed_dog, (512, 512))
new_image = Image.fromarray((reshaped_dog).astype(np.uint8))
old_image = Image.open('dog.jpg')


# print(f'Pixels in new image {pixelcount(new_image)}')
# print(f'Pixels in old image {pixelcount(old_image)}')

# are_images_equal(new_image, old_image)

# compare(old_image,new_image)
print(f'\nThe shape of the reconstructed tensor is: {reconstructed_dog.shape}')
reshaped_dog = np.reshape(reconstructed_dog*255, (512, 512))
new_image = Image.fromarray((reshaped_dog).astype(np.uint8))
old_image = Image.open('dog.jpg')

# print(reshaped_dog)

print(f'Pixels in new image {pixelcount(new_image)}')
print(f'Pixels in old image {pixelcount(old_image)}')

are_images_equal(new_image, old_image)
compare(old_image,new_image)

"""
sitek(tt,k), d:#cores -1
    for i = d:-1:k+1
    C = tt[i]
    C = mode 1 unfold (C)
    [Q,R] = qr(np.transpose(C)
    tt[i] = np.reshape(Q}
    tt[i-1] mode3product(tt[i-1],R)

mode 2 permute before reshape
"""