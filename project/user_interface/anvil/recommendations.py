from internal_image_tt import internal_image_tensor_train, internal_process_image
import matplotlib.pyplot as plt



def recommend(epsilon, img):
    epsilons = []
    accuracies = []
    entries = []
    single = epsilon / 100
    start = epsilon-(5*single)
    for i in range(10):
        epsilons.append(start)
        start += single
    for ep in epsilons:
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