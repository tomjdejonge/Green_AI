import numpy as np
from TT_SVD_1 import tensortrain, tt_reconstruction, border
from time import process_time
from PIL import Image
import matplotlib.pyplot as plt


epsilons = np.arange(0.01, 1, 0.01)
l = len(epsilons)
tt_elements = np.zeros(l)
total_elements = np.zeros(l)
percentage = []
t1 = []
t2 = []

img = Image.open('C:/Users/tommo/Downloads/dog.jpg.jpeg')
img = np.asarray(img)

def accuracy(image, epsilon):
    g, d, r, n = tensortrain(image, epsilon)
    reconstructed = tt_reconstruction(g, d, r, n)
    bordered = border(np.array(reconstructed), 0, 255, p=True)
    error = np.sum(np.absolute(np.subtract(image, bordered)))
    total = np.sum(image)
    error_percentage = error / total
    return error_percentage

error = accuracy(img, 0.1)

c = 0
for k in epsilons:
    t_start = process_time()
    g, d, r, _ = tensortrain(img, k)
    t_stop = process_time()
    t1.append(t_stop-t_start)
    for i in range(d+1):
        tt_elements[i] = np.size(g[i])
    total_elements[c] = sum(tt_elements)
    percentage.append(total_elements[c] / 786432)
    c += 1

figure, axis = plt.subplots(3)

axis[0].plot(epsilons, percentage)
axis[0].set_title("Epsilon vs Percentage of Total Entries")
axis[1].plot(total_elements, t1)
axis[1].set_title("No of Elements vs Calculation time")
axis[2].plot(percentage, t1)
axis[2].set_title("Percentage of entries vs Calculation time")

plt.show()