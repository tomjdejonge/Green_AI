import matplotlib.image as image
import numpy as np
img=image.imread('dog.jpg')
print('The Shape of the image is:',img.shape)
print('The image as array is:')
print(img)
reshape = np.reshape (img, (512,1536))
svd = np.linalg.svd(reshape,compute_uv=True)
sv, u, d = svd
print(f'sv = {sv.shape}')
print(f'u = {u.shape}')
print(f'd = {d.shape}')