import numpy as np
from PIL import Image
from scipy import linalg


dog = Image.open('dog.jpg')
dog_array = np.asarray(dog)
print(dog_array)