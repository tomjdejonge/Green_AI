import matplotlib.image as image
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from scipy.linalg import svd

img = image.imread('dog.jpg')
print('The shape of the image is:', img.shape)

# Reshape image
reshaped_img = np.reshape(img, (512, 1536))
print('The shape of the reshaped image is:', reshaped_img.shape)

# Perform SVD
# u, s, vh = np.linalg.svd(reshaped_img, full_matrices=True)
# print(f'The shape of u is: {u.shape}')
# print(f'The shape of s is: {s.shape}')
# print(f'The shape of vh is: {vh.shape}')


# reconstruction = u @ s @ vh
# np.array_equal(reconstruction, reshaped_img)
