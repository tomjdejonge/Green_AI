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
    deconstruction_time = float(np.round(t_stop - t_start, 3))
    tt_elements = []
    for i in range(d + 1):
        tt_elements.append(np.size(g[i]))
    total_elements = sum(tt_elements)
    percentage = float(np.round((total_elements / 786432) * 100, 3))
    reconstructed = tt_reconstruction(g, d, r, n)
    new_image = border(np.array(reconstructed), 0, 255, p=True)
    error = np.sum(np.absolute(np.subtract(img, new_image)))
    total = np.sum(new_image)
    accuracy_percentage = float(np.round((1 - (error / total)), 3))
    final_image = Image.fromarray(new_image.astype(np.uint8), 'RGB')
    bs = io.BytesIO()
    name = 'final_image'
    final_image.save(bs, format="JPEG")
    final = anvil.BlobMedia("image/jpeg", bs.getvalue(), name=name)
    return g, d, r, n, final, deconstruction_time, percentage, total_elements, accuracy_percentage


@anvil.server.callable
def import_csv_data(file):
    with anvil.media.TempFile(file) as file_name:
        # df = pd.read_csv(file_name)
        return file_name


@anvil.server.callable
def import_csv_data_2_calculate(file):
    df = pd.read_csv(io.BytesIO(file.get_bytes()), header=0)
    naive_accuracy, naive_sd, naive_time = svm(df)
    return naive_accuracy, naive_sd, naive_time


# @anvil.server.callable
# def store_data(file):
#     with anvil.media.TempFile(file) as file_name:
#         if file.content_type == 'text/csv':
#             df = pd.read_csv(file_name)
#         else:
#             df = pd.read_excel(file_name)
#         for d in df.to_dict(orient="records"):
#             # d is now a dict of {columnname -> value} for this row
#             # We use Python's **kwargs syntax to pass the whole dict as
#             # keyword arguments
#             app_tables.data.add_row(**d)


# io.BytesIO(image.get_bytes())

# @anvil.server.callable
# def svm(dataset):
#     # Read the dataset
#     dataset = pd.read_csv(dataset, header=0)
#
#     colnames = list(dataset.iloc[:, -1].unique())
#
#     # print(dataset)
#     # Encoding the categorical column
#     dataset = dataset.replace({"class": {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}})
#     X = dataset.iloc[:, :-1]
#     y = dataset.iloc[:, -1].values
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#
#     # Create the SVM model
#     classifier = SVC(kernel='linear', random_state=0)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#
#     cm = confusion_matrix(y_test, y_pred)
#     print(cm)
#
#     accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
#     print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
#     print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
#
#     print(time.process_time())


@anvil.server.callable
def clear_table():
    app_tables.table_0.delete_all_rows()


anvil.server.wait_forever()
