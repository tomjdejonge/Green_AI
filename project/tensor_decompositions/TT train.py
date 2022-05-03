import matplotlib.image as image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image

img=image.imread('dog.jpg')
dog = img.reshape(4,4,4,4,4,4,4,4,4,3)

def TT(pic):
    if isinstance(pic, np.ndarray):
        x = pic
        threshold = 0
        max_rank = np.infty

        order = len(x.shape) // 2
        row_dims = x.shape[:order]
        col_dims = x.shape[order:]
        ranks = [1] * (order + 1)
        cores = []

        p = [order * j + i for i in range(order) for j in range(2)]
        y = np.transpose(x, p).copy()


        #iterate
        for i in range(order-1):
            m = ranks[i] * row_dims[i] * col_dims[i]
            n = np.prod(row_dims[i + 1:]) * np.prod(col_dims[i + 1:])
            y = np.reshape(y, [m, n])

            [u, s, v] = np.linalg.svd(y, full_matrices=False, compute_uv=True)

            if threshold != 0:
                indices = np.where(s / s[0] > threshold)[0]
                u = u[:, indices]
                s = s[indices]
                v = v[indices, :]
            if max_rank != np.infty:
                u = u[:, :np.minimum(u.shape[1], max_rank)]
                s = s[:np.minimum(s.shape[0], max_rank)]
                v = v[:np.minimum(v.shape[0], max_rank), :]

            ranks[i + 1] = u.shape[1]
            cores.append(np.reshape(u, [ranks[i], row_dims[i], col_dims[i], ranks[i + 1]]))

            y = np.diag(s).dot(v)

        cores.append(np.reshape(y, [ranks[-2], row_dims[-1], col_dims[-1], 1]))

        print('\n'
              'Tensor train with order    = {d}, \n'
              '                  row_dims = {m}, \n'
              '                  col_dims = {n}, \n'
              '                  ranks    = {r}'.format(d=order, m=row_dims, n=col_dims, r=ranks))

        return cores
    else:
        print('wrong dimensions')

def full(list,x):
    order = len(x.shape) // 2
    row_dims = x.shape[:order]
    col_dims = x.shape[order:]
    ranks = [1] * (order + 1)

    #split
    full = list[0].reshape(row_dims[0] * col_dims[0], ranks[1])

    for i in range (1, order):
        full = full.dot(cores[i].reshape(ranks[i], row_dims[i]*col_dims[i]*ranks[i+1]))

        full = full.reshape(np.prod(row_dims[:i + 1]) * np.prod(col_dims[:i + 1]),ranks[i + 1])

    return full



print(TT(dog))

full(TT(dog),dog)




#
#
# recon_reshaped_image = Image.fromarray((res* 255).astype(np.uint8))
# recon_reshaped_image.show()







