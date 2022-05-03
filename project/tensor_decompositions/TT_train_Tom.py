from PIL import Image
import numpy as np


# Tensor Train decomposition for a square image
def tt_square_image(image):
    # Load the image and convert image to numpy array
    image = Image.open(image)
    img_array = np.asarray(image)
    print(f'The shape of the original image is: {img_array.shape}')
    # Reshape
    img_array = np.reshape(img_array, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
    print('The shape of the initially reshaped image is:', img_array.shape)
    no_dim = len(img_array.shape)
    print(f'The number of dimensions is now: {no_dim}')
    U = []
    S = []
    VH = []
    for i in range(no_dim - 1):
        u, s, vh = svd(img_array)
        U.append(u)
        S.append(s)
        VH.append(vh)
        img_array = s
        print(s.shape)
    print(f'Length of U = {len(U)}, length of S = {len(S)}, length of VH = {len(VH)}')
    return U, S, VH


# Single Value Decomposition
def svd(image):
    u, s, vh = np.linalg.svd(image, full_matrices=True, compute_uv=True)
    return u, s, vh


tt_square_image('dog.jpg')

# # Reconstruction
# recon = np.matmul(u * s[..., None, :], vh)
#
# # Reshape
# recon_reshaped = np.reshape(recon, (512, 512, 3))
# print('The shape of the final reshaped image is:', recon_reshaped.shape)
# print(type(recon_reshaped))
# recon_reshaped_image = Image.fromarray((recon_reshaped * 255).astype(np.uint8))
# recon_reshaped_image.show()
