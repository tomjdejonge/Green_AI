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

        #
        # x = np.matmul(ut[0], ut[1])
        # y = np.matmul(ut[2], ut[3])
        return cores
    else:
        return DimensionError

def mul(list):
    #split
    leng = len(list)//2

    first = (list[:leng])
    last = (list[(leng):])
    # print(first.shape, last.shape)
    x = np.matmul(first[0],first[1].Transposed)
    y = np.matmul(last[0],last[1])
    return  np.matmul(x,y) #(first[0]*first[1])*(last[0]*last[1])


TT(dog)



#
#
# recon_reshaped_image = Image.fromarray((res* 255).astype(np.uint8))
# recon_reshaped_image.show()







