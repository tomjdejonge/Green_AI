import matplotlib.image as image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image

img=image.imread('dog.jpg')
dog = img.reshape(4,4,4,4,4,4,4,4,4,3)

order = len(dog.shape)//2
row_dims = dog.shape[:order]
col_dims = dog.shape[order:]
ranks = [1] * (order +1)
cores = []


def svd(pic):
    return np.linalg.svd(pic, full_matrices=False, compute_uv=True)

def TT(pic):
    # variables
    ut = []
    st = []
    vht = []

    order = len(pic.shape) // 2
    row_dims = pic.shape[:order]
    col_dims = pic.shape[order:]
    ranks = [1] * (order + 1)
    cores = []
    # calculate delta
    #delta = (eps/(np.sqrt(d-1))) * pic.length()

    p = [order * j + i for i in range(order) for j in range(2)]
    y = np.transpose(pic, p).copy()

    #iterate
    for i in range(order-1):
        core = pic[i]

        [u, s, v] = np.linalg.svd(core, full_matrices=False, compute_uv=True)

        ut.append(u)
        st.append(s)
        vht.append(v)
        cores.append(s)

    #
    # x = np.matmul(ut[0], ut[1])
    # y = np.matmul(ut[2], ut[3])

    return cores

def mul(list):
    #split
    leng = len(list)//2

    first = list[:leng]
    last = list[(leng):]
    print(len(first))
    x = np.matmul(first[0],first[1])
    y = np.matmul(last[0],last[1])
    return  np.matmul(x,y) #(first[0]*first[1])*(last[0]*last[1])


res = TT(dog)

print(mul(TT(dog)))

#
#
# recon_reshaped_image = Image.fromarray((res* 255).astype(np.uint8))
# recon_reshaped_image.show()







