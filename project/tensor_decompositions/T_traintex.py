# from PIL import Image
# from scipy import linalg
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
#  ## ucidatasets
# def tt_decomposition(img, epsilon=0.01):
#     # Load the image and convert image to correct numpy array
#     img = Image.open(img)
#     x = np.asarray(img)
#     # print(x)
#     print(f'The shape of the original image is: {x.shape}')
#     C = np.reshape(x, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
#     C1 = C
#     print('The shape of the initially reshaped image is:', C.shape)
#     n = [4, 4, 4, 4, 4, 4, 4, 4, 4, 3]
#     d = np.ndim(C)
#     cores = [0] * (d)
#     delta1 = epsilon / (np.sqrt(d-1)) * np.linalg.norm(x)
#     delta =  epsilon / (np.sqrt(d-1)) * fro(x)
#     delta2 = epsilon / (np.sqrt(d-1)) * frob(x)
#     # print(f' delta = {delta}, delta1 = {delta1} , delta2 = {delta2}')
#     r = [0] * (d)
#     r[0] = 1
#
#     for i in range(d-1):
#         m = int(r[i] * n[i])                                   # r_(k-1)*n_k
#         b = int(C.size/m)                                   # numel(C)/r_(k-1)*n_k    or  n = row_dims[i + 1:]) * np.prod(col_dims[i + 1:]
#         print(f'C.size{i} = {C.size}')
#         C = np.reshape(C, [m, b])
#         terror = 0
#
#         [u, s, v] = linalg.svd(C, full_matrices=False)          #(u @ np.diag(s) @ v).astype(int)
#
#         rk = 0
#         s = np.diag(s)
#         error = linalg.norm((s[rk+1:]))
#         print(f'{i,u.shape,s.shape,v.shape}')
#         if epsilon != 0:
#             while error > delta:
#                 rk +=1
#                 error = linalg.norm(s[rk+1:])
#             # print(f'error = {error}')
#             # print(f'r{i} = {rk}')
#         else:
#             rk = 1
#
#         r[i+1] = rk
#         terror += error ** 2
#         # cores.append(np.reshpa)
#         cores[i] = (np.reshape(u[:,:r[i+1]], [r[i], n[i], r[i + 1]]))
#         # print(f's{i} = {s[0:r[i+1],0:r[i+1]]}')
#         # print(f'v{i} = {v[:,0:r[i+1]]}')
#         # # print(f'indices = {r[i+1]}')
#         # print(f'{i, s[:r[i+1],:r[i+1]].shape, (v[:r[i+1],:]).shape}')
#         C = (s[:r[i+1],:r[i+1]])@(v[:r[i+1],:])
#
#
#     cores[-1] = (np.reshape(C, [r[-1], n[-1], 1]))       #C, [r[-2], n[-1], n[-1], 1]))
#     rerror = np.sqrt(terror)/np.linalg.norm(x)
#     # print(f'cores = {np.linalg.norm(cores[0])}, original tensor = {np.linalg.norm(x)}')
#     for i in range(d):
#         print(i, np.linalg.norm(cores[i]))
#     print(fro(x))
#     # print('coress=',cores[-7])
#     print("\n"
#           "Tensor train created with order    = {d}, \n"
#           "                  row_dims = {m}, \n"
#           "                  col_dims = {n}, \n"
#           "                  ranks    = {r}  \n"
#           "                  relative error   = {t}"    .format(d=d, m=n, n=n, r=r, t=rerror))
#     return cores, n, r, d
#
#
# # Single Value Decomposition
# def svd(x):
#     u, s, v = linalg.svd(x, full_matrices=False)
#     return u, s, v
#
# def fro(x):
#     count = 0
#     res = 0
#     for i in range(len(x)):
#         for j in range(len(x)):
#             for k in range(3):
#                 res += pow(x[i][j][k], 2)
#                 count += 1
#     return round(np.sqrt(res),4)
#
# def frob(tensor):
#     tensor = tensor.flatten()
#     return np.sqrt(np.matmul(tensor.transpose(), tensor))
#
# dog_tensor = tt_decomposition('dog.jpg')
# cores, n, r, d = dog_tensor
#
#
# # Tensor Train Reconstruction 2
# def tt_reconstruction_2(cores, n, r, d):
#
#         # reshape first core
#     full_tensor = cores[0].reshape(n[0], r[1])
#
#     # print(f'd={d}')
#     for i in range(1, (d//2)+2):
#         # contract full_tensor with next TT core and reshape
#         full_tensor = full_tensor.dot(cores[i].reshape(r[i], n[i] * r[i+1]))
#         full_tensor = full_tensor.reshape((np.prod(n[:i + 1])), r[i+1])
#
#     print(f'n = {n[:-1]}')
#     q = [2 * i for i in range(d//2)]
#     full_tensor = full_tensor.reshape(n).transpose(q)
#
#     return full_tensor
#
# def compare(image1, image2):
#     f = plt.figure()
#     f.add_subplot(1,2, 1)
#     plt.imshow(image1,interpolation='nearest')
#     plt.axis('off')
#     f.add_subplot(1,2, 2)
#     plt.imshow(image2,interpolation='nearest')
#     plt.axis('off')
#     plt.show(block=True)
#
# def are_images_equal(im1, im2):
#     if list(im1.getdata()) == list(im2.getdata()):
#         print("\nThe images are identical")
#     else:
#         print("\nThe images are different")
#
# def pixelcount(img):
#     count = 0
#     for y in range(img.height):
#         for x in range(img.width):
#             count += 1
#
#     return count
#
# reconstructed_dog = tt_reconstruction_2(cores, n,r,d)
#
# print(f'\nThe shape of the reconstructed tensor is: {reconstructed_dog.shape}')
# reshaped_dog = np.reshape(reconstructed_dog, (512, 512))
# new_image = Image.fromarray((reshaped_dog).astype(np.uint8))
# old_image = Image.open('dog.jpg')
#
#
# # print(f'Pixels in new image {pixelcount(new_image)}')
# # print(f'Pixels in old image {pixelcount(old_image)}')
#
# # are_images_equal(new_image, old_image)
#
# # compare(old_image,new_image)
# print(f'\nThe shape of the reconstructed tensor is: {reconstructed_dog.shape}')
# reshaped_dog = np.reshape(reconstructed_dog*255, (512, 512))
# new_image = Image.fromarray((reshaped_dog).astype(np.uint8))
# old_image = Image.open('dog.jpg')
#
# # print(reshaped_dog)
#
# print(f'Pixels in new image {pixelcount(new_image)}')
# print(f'Pixels in old image {pixelcount(old_image)}')
#
# are_images_equal(new_image, old_image)
# compare(old_image,new_image)
#
# """
# sitek(tt,k), d:#cores -1
#     for i = d:-1:k+1
#     C = tt[i]
#     C = mode 1 unfold (C)
#     [Q,R] = qr(np.transpose(C)
#     tt[i] = np.reshape(Q}
#     tt[i-1] mode3product(tt[i-1],R)
#
# mode 2 permute before reshape
# """
# from PIL import Image
# from scipy import linalg
# import numpy as np
# import torch
# # import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
import numpy as np
import tensorly as ty
# import matplotlib.pyplot as plt

img = Image.open('dog.jpg')
p = np.asarray(img)
a = p

d = 3
r0 = 1
r3 = r0
epsilon = 0.2
delta = (epsilon / np.sqrt(d-1)) * np.linalg.norm(a)

n = a.shape
r = np.zeros(d+1)
r[0] = r0
r[d] = r3
g = []

c = a

for k in range(d-1):
    m = int(r[k] * n[k])  # r_(k-1)*n_k
    b = int(c.size / m)  # numel(C)/r_(k-1)*n_k
    c = np.reshape(c, [m, b])
    # print(c.shape)
    [U, S, V] = linalg.svd(c, full_matrices=False)
    V = V.transpose()
    s = np.diag(S)
    rank = 0
    # print(S[rank])
    error = np.linalg.norm(s[rank])
    # print(error)
    while error > delta:
        rank += 1
        error = np.linalg.norm(s[rank+1:])
    r[k+1] = rank + 1
    # print(int(r[k+1]))
    # print(int(n[k]))
    # print(int(r[k]))
    # print((U[:, :int(r[k+1])]).shape)
    g.append(np.reshape(U[:, :int(r[k+1])], [int(r[k]), int(n[k]), int(r[k+1])]))
    # print(np.linalg.norm(g[k]))
    # print(S.shape)
    p_1 = s[:int(r[k + 1]), :int(r[k + 1])]
    # print(p_1)
    p_2 = V[:, :int(r[k + 1])]
    # print(p_2)
    c = p_1 @ p_2.transpose()
    print(c.shape)
# print(r[d-1])
# print(n[d-1])
# print(r[d])
g.append(np.reshape(c, (int(r[d-1]), int(n[d-1]), int(r[d]))))
# print(g[0].shape)
# print(g[1].shape)
# print(g[2].shape)
g1 = np.transpose(g[0], [1, 2, 0])
# g1 = np.squeeze(g1)
# print(g1.shape)
g2 = g[1]
# print(g2.shape)
g3 = np.transpose(g[d-1])
g3 = np.squeeze(g3)
# print(g3.shape)

# g1 = np.moveaxis(g[0],[0,1,2],[1,2,0])
# g2 = g[1]
# g3 = g[2].transpose()

# print(g1.shape, g2.shape, g3.shape)

# I = np.tensordot(g2, g3, 3)
# B = np.matmul(I, g3)

x = ty.tenalg.mode_dot(g2, g3, 2)

print(g1.shape, x.shape)

B = []
for i in range(0,3):

    res = np.matmul(np.transpose(x[:,:,i]),g1)
    B.append(res)
    print(B[0].shape)
B = np.array(B)
B = np.squeeze(B)
B = np.transpose(B)
print( B.shape)

new_image = Image.fromarray((B).astype(np.uint8))


def compare(image1, image2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image1,interpolation='nearest')
    plt.axis('off')
    f.add_subplot(1,2, 2)
    plt.imshow(image2,interpolation='nearest')
    plt.axis('off')
    plt.show(block=True)

compare(new_image, new_image)