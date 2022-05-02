import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

MAX_RANK = 50
FNAME = 'dog.jpg'

image = Image.open(FNAME).convert("L")
img_mat = np.asarray(image)
print(img_mat.shape)
print(FNAME.shape)
#
# U, s, V = np.linalg.svd(img_mat, full_matrices=True)
# s = np.diag(s)
#
# for k in range(MAX_RANK + 1):
#     approx = U[:, :k] @ s[0:k, :k] @ V[:k, :]
#     img = plt.imshow(approx, cmap='gray')
#     plt.title(f'SVD approximation with degree of {k}')
#     plt.plot()
#     pause_length = 0.0001 if k < MAX_RANK else 5
#     plt.pause(pause_length)
#     plt.clf()
