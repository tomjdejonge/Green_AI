# -*- coding: utf-8 -*-
"""
code from:https://medium.com/@pinnzonandres/iris-classification-with-svm-on-python-c1b6e833522c
"""
import pandas as pd
import time as time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from time import process_time

iris = "C:/Users/tommo/Downloads/iris.csv"


def svm(dataset, miter=2, split=0.33):
    t_start = process_time()
    # Encoding the categorical column
    # dataset = dataset.replace({"class": {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}})
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=0)

    # Create the SVM model
    classifier = SVC(kernel='linear', random_state=0, max_iter=miter)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    naive_accuracy = (accuracies.mean() * 100)
    naive_sd = (accuracies.std() * 100)
    t_stop = process_time()
    naive_time = float(t_stop - t_start)

    print(f"Accuracy: {naive_accuracy} %")
    print(f"Standard Deviation: {naive_sd} %")
    print(f"Time: {naive_time} seconds")

    return naive_accuracy, naive_sd, naive_time