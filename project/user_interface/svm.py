# -*- coding: utf-8 -*-
"""
code from:https://medium.com/@pinnzonandres/iris-classification-with-svm-on-python-c1b6e833522c
"""
import pandas as pd
import time as time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

#Define the col names
"/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/winequality-white.csv"

#Read the dataset
# dataset = pd.read_csv("/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/Iris.csv", header = 0 )
dataset = pd.read_csv('/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/pima-indians-diabetes.csv', header = 0)
# dataset = pd.read_csv('"/Users/Tex/PycharmProjects/Green_AI/project/tensor_decompositions/winequality-white.csv"', header = 0)


def svm(dset, miter=2, split=0.33):
#Encoding the categorical column
# dataset = dataset.replace({"class":  {"Iris-setosa":1,"Iris-versicolor":2, "Iris-virginica":3}})
    X = dset.iloc[:,:-1]
    y = dset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split, random_state = 0)

    #Create the SVM model
    classifier = SVC(kernel = 'linear', random_state = 0, max_iter=miter)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    acc = format(accuracies.mean()*100)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
    tijd = time.process_time()
    print(time.process_time())
    return(acc,tijd)

print(svm(dataset))