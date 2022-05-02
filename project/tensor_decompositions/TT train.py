import matplotlib.image as image
import numpy as np
from skimage import color
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt

img=image.imread('dog.jpg')
dog = img.reshape(4,4,4,4,4,4,4,4,4,3)
print(dog.shape)

def svd(pic):
    svd = np.linalg.svd(pic, full_matrices=True, compute_uv=True)
    u, s, vh = svd
    return u, s, vh

def TT(pic):
    # variables
    ut = []
    st = []
    vht = []
    # calculate delta
    #delta = (eps/(np.sqrt(d-1))) * pic.length()

    #iterate
    for i in range(8):
        u, s, vh = svd(pic)
        ut.append(u)
        st.append(s)
        vht.append(vh)
        pic = s
        print(s.shape)
    #print(st)

    return None

print(f'hoi, {TT(dog)}')







