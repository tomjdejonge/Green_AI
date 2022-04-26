import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face # for the raccoon’s face image
from scipy.ndimage import zoom # to zoom the image represented in the form of an array
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from math import ceil

img = tl.tensor(zoom(face(), (0.3, 0.3, 0.3)), dtype='float64') # turn image into tensor


def to_img(tensor):
    image = tl.to_numpy(tensor) # convert tensor to np array
    image -= image.min()
    image /= image.max()
    image *= 255
    return image.astype(np.uint8)


# define ranks of decomposition
cp_rank = 25
tucker_rank = [100, 100, 2]

# perform CP decomposition
weights, factors = parafac(img, rank=cp_rank, init='random', tol=10e-6)
cp_rec = tl.cp_to_tensor((weights, factors)) # reconstruct the image from the factors

# perform Tucker decomposition
core, factors = tucker(img, rank=tucker_rank, init='random', tol=10e-5, random_state=12345)
tucker_rec = tl.tucker_to_tensor((core, factors)) # convert the tucker tensor into a full tensor

# plot original image
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1) # arguments represent (number of rows, number of columns, index)
ax.set_axis_off() # turn the X and Y axes off
ax.imshow(to_img(img))  # display the plot
ax.set_title('Original Image') # title of the plot

# plot compressed image using CP and Tucker decompositions
ax = fig.add_subplot(1, 3, 2)
ax.set_axis_off()
ax.imshow(to_img(cp_rec))
ax.set_title('CP decomposition')
ax = fig.add_subplot(1, 3, 3)
ax.set_axis_off()
ax.imshow(to_img(tucker_rec))
ax.set_title('Tucker Decomposition')
plt.tight_layout() # adjust the padding between and around the subplots
plt.show() # display the plots