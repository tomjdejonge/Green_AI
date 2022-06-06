import numpy as np
from TT_SVD_1 import tensortrain, tt_reconstruction, border
from PIL import Image


img = Image.open('C:/Users/tommo/Downloads/dog.jpg.jpeg')
img = np.asarray(img)

def accuracy(image, epsilon):
    g, d, r, n = tensortrain(image, epsilon)
    reconstructed = tt_reconstruction(g, d, r, n)
    bordered = border(np.array(reconstructed), 0, 255, p=True)
    error = np.sum(np.absolute(np.subtract(image, bordered)))
    total = np.sum(image)
    error_percentage = error / total
    print(error_percentage)
    # new_image = Image.fromarray(bordered.astype(np.uint8), 'RGB')


accuracy(img, 0.1)