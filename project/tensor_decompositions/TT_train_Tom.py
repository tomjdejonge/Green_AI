from PIL import Image
from numpy import asarray
import numpy as np

# load the image and convert image to numpy array
image = Image.open('dog.jpg')
img = asarray(image)
print(f'The shape of the original image is: {img.shape}')

# Reshape
img = np.reshape(img, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
print('The shape of the initially reshaped image is:', img.shape)

# SVD
no_dim = len(img.shape)

u, s, vh = np.linalg.svd(img, full_matrices=False)

# count = 0
# U = []
# S = []
# VH = []
#
# while count < no_dim:
#     u, s, vh = np.linalg.svd(img, full_matrices=False)
#     U.append(u)
#     S.append(s)
#     VH.append(vh)
#     img = s
#     no_dim -= 1
#     count += 1
#     print(f'Dim of img after SVD run no {count}: {img.shape}')
#
# print(f'The shapes of: U = {U.shape}, S = {S.shape}, VH = {VH.shape}')

# Reconstruction
recon = np.matmul(u * s[..., None, :], vh)

# Reshape
recon_reshaped = np.reshape(recon, (512, 512, 3))
print('The shape of the final reshaped image is:', recon_reshaped.shape)
print(type(recon_reshaped))
recon_reshaped_image = Image.fromarray((recon_reshaped * 255).astype(np.uint8))
recon_reshaped_image.show()
