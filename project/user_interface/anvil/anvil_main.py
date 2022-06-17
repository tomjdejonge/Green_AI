import anvil.server
import anvil.media
import io
from PIL import Image
from TT_SVD_1 import tensortrain, tt_reconstruction, border
import numpy as np
import anvil.tables as tables
from anvil.tables import app_tables
from time import process_time
import pandas as pd
import time as time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from svm_orig import svm
from als_orig import t_test, datareader
from internal_image_tt import internal_image_tensor_train
import matplotlib.pyplot as plt
import anvil.mpl_util
from naive import naive

anvil.server.connect('DH4R5MH3DTKYNYBPF63XKZLV-UHSXWMNTP7IBV2RQ')


@anvil.server.callable
def generate_iris():
    iris = anvil.media.from_file('iris.csv', 'csv', name='Iris.csv')
    return iris


@anvil.server.callable
def generate_dog():
    dog = anvil.media.from_file('C:/Users/tommo/PycharmProjects/Green_AI/project/user_interface/anvil/images/dog.jpg',
                                'csv', name='dog.jpg')
    return dog


@anvil.server.callable
def process_image(image):
    image_object = Image.open(io.BytesIO(image.get_bytes()))
    image = np.asarray(image_object)
    return image


@anvil.server.callable
def image_tensor_train(img, epsilon=0.1):
    t_start = process_time()
    g, d, r, n = tensortrain(img, epsilon)
    t_stop = process_time()
    deconstruction_time = float(t_stop - t_start)
    tt_elements = []
    for i in range(d + 1):
        tt_elements.append(np.size(g[i]))
    total_elements = sum(tt_elements)
    percentage = float(np.round((total_elements / 786432) * 100, 3))
    reconstructed = tt_reconstruction(g, d, r, n)
    new_image = border(np.array(reconstructed), 0, 255, p=True)
    error = np.sum(np.absolute(np.subtract(img, new_image)))
    total = np.sum(new_image)
    accuracy_percentage = float((1 - (error / total)) * 100)
    final_image = Image.fromarray(new_image.astype(np.uint8), 'RGB')
    bs = io.BytesIO()
    name = 'final_image'
    final_image.save(bs, format="JPEG")
    final = anvil.BlobMedia("image/jpeg", bs.getvalue(), name=name)
    return g, d, r, n, final, deconstruction_time, percentage, total_elements, accuracy_percentage


@anvil.server.callable
def plot_ep_vs_entries(img, epsilon):
    epsilons = list(np.arange(0.01, 0.501, 0.01))
    accuracies = []
    entries = []
    single = epsilon / 100
    # start = epsilon - (5 * single)
    # for i in range(10):
    #     epsilons.append(start)
    #     start += single
    for ep in epsilons:
        print(f'calculating ep {ep}')
        g, d, r, n, final, deconstruction_time, percentage, total_elements, accuracy_percentage = internal_image_tensor_train(
            img, ep)
        accuracies.append(accuracy_percentage)
        entries.append(total_elements / 786432)
    ind = epsilons.index(epsilon)
    #
    # axs[1].plot(epsilons, entries)
    # axs[1].plot(epsilons[ind], accuracies[ind], 'r*')
    # anvil.mpl_util.plot_image()
    return epsilons, entries, accuracies, ind


@anvil.server.callable
def ep_vs_ent(epsilons, entries, ind):
    fig, ax = plt.subplots()
    ax.plot(epsilons, entries)
    ax.plot(epsilons[ind], entries[ind], 'r*')
    ax.set(xlabel='Accuracy Epsilon', ylabel='Entries',
           title='Accuracy Epsilon vs Entries')
    ax.grid()
    fig.savefig("ent.png")
    return anvil.mpl_util.plot_image()


@anvil.server.callable
def ep_vs_acc(epsilons, accuracy, ind):
    fig, ax = plt.subplots()
    ax.plot(epsilons, accuracy)
    ax.plot(epsilons[ind], accuracy[ind], 'r*')
    ax.set(xlabel='Accuracy Epsilon', ylabel='Relative Error',
           title='Accuracy Epsilon vs Relative Error')
    ax.grid()
    fig.savefig("acc.png")
    return anvil.mpl_util.plot_image()


@anvil.server.callable
def import_csv_data_2_calculate(file, miter, split, I=1):
    df = pd.read_csv(io.BytesIO(file.get_bytes()), header=0)
    df_2 = datareader(file)
    naive_accuracy, naive_sd, naive_time = svm(df, miter, split)
    new_accuracy, new_time = t_test(df, I, miter, split)
    normal_accuracy, normal_time = naive(df_2, split)
    return naive_accuracy, naive_sd, naive_time, new_accuracy, new_time, normal_accuracy, normal_time


@anvil.server.callable
def clear_table():
    app_tables.table_0.delete_all_rows()


anvil.server.wait_forever()
