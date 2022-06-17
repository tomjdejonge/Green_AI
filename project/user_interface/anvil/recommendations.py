from internal_image_tt import internal_image_tensor_train, internal_process_image
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal as D



def recommend(epsilon, img):
    epsilons = [np.arange(D("0.01"), D("0.51"), D("0.01"))]
    accuracies = []
    entries = []
    single = 0.01
    start = 0.01
    # for i in range(50):
    #     # epsilons.append(start)
    #     start += single
    for ep in epsilons:
        ep = round(ep, 2)
        print(f'calculating ep {ep}')
        g, d, r, n, final, deconstruction_time, percentage, total_elements, accuracy_percentage = internal_image_tensor_train(img, ep)
        accuracies.append(accuracy_percentage)
        entries.append(total_elements)
    return epsilons, accuracies, entries


# dog = internal_process_image("C:/Users/tommo/Downloads/dog.jpg")
# epsilons, accuracies, total_elements = recommend(0.05, dog)
#
# plt.plot(total_elements, accuracies)
# plt.show()