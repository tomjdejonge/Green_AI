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
from als_orig import t_test


def internal_process_image(image):
    image_object = Image.open(image)
    image = np.asarray(image_object)
    return image


def internal_image_tensor_train(img, epsilon=0.1):
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
    accuracy_percentage = float(np.round((1 - (error / total)) * 100, 3))
    final_image = Image.fromarray(new_image.astype(np.uint8), 'RGB')
    bs = io.BytesIO()
    name = 'final_image'
    final_image.save(bs, format="JPEG")
    final = anvil.BlobMedia("image/jpeg", bs.getvalue(), name=name)
    return g, d, r, n, final, deconstruction_time, percentage, total_elements, accuracy_percentage