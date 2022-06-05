import numpy as np
from PIL import Image


epsilons = np.arange(0.01, 1, 0.01)
l = len(epsilons)
tt_elements = np.zeros(l)
total_elements = np.zeros(l)
percentage = []
t1 = []
t2 = []

img = Image.open('C:/Users/tommo/Downloads/dog.jpg.jpeg')
img = np.asarray(img)

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