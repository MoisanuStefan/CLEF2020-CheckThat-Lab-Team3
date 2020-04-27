import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from os import path
from joblib import dump, load
from svm_exceptions import *


# Argument defaults:
# kernel = 'rbf'
# degree = 3
# C = 1.0
# random_state = None
# gamma = 'scale'
# coef0 = 0.0
# shrinking = True
# verbose = False
def get_fitted_svclassifier(X_train, y_train, kernel, degree, C, random_state, gamma, coef0, shrinking, verbose):
    svclassifier = SVC(kernel=kernel, degree=degree, C=C, random_state=random_state, gamma=gamma, coef0=coef0,
                       shrinking=shrinking, verbose=verbose)
    svclassifier.fit(X_train, y_train)  # actual training
    return svclassifier


def serialization(path_to_object, svclassifier):
    if not isinstance(svclassifier, SVC):
        raise IsNotSVCException('Object passed as parameter is not an instance of SVC')
    elif path.exists(path_to_object):
        raise PathExistsException('File already exists')
    else:
        dump(svclassifier, path_to_object)


def deserializer(path_to_object):
    if path_to_object[-7:] != '.joblib':
        raise WrongExtensionException('Type of input file is not \'.joblib\'')
    elif not path.exists(path_to_object):
        raise PathNotExistsException('File not exists')
    svclassifier = load(path_to_object)
    return svclassifier


def predict_y(svclassifier, X_test):
    y_pred = svclassifier.predict(X_test)
    return y_pred


def svcClassify1():
    svclassifier = get_fitted_svclassifier(X_train, y_train, 'linear', 3, 1, 5, 'scale', 0.0, True,
                                           False)  # basic default

    y_pred = predict_y(svclassifier, X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'Class\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')


def svcClassify2():
    path_to_model = 'model.joblib'

    if not path.exists(path_to_model):
        svclassifier = get_fitted_svclassifier(X_train, y_train, 'poly', 5, 1.9, 5, 'auto', 0.2, True,
                                               False)  # the precision is almost perfect for both 0 and 1
        serialization(path_to_model, svclassifier)
    else:
        svclassifier = deserializer(path_to_model)

    y_pred = predict_y(svclassifier, X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'Class\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')


def svcClassify3():
    svclassifier = get_fitted_svclassifier(X_train, y_train, 'poly', 4, 1.5, 3, 'auto', 0.1, False,
                                           False)  # lower precision for both

    y_pred = predict_y(svclassifier, X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'Class\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')


def svcClassify4():
    svclassifier = get_fitted_svclassifier(X_train, y_train, 'linear', 1, 1.7, 2, 'scale', 0.06, True,
                                           False)  # the accuracy is not the greatest

    y_pred = predict_y(svclassifier, X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'Class\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')


def svcClassify5():
    svclassifier = get_fitted_svclassifier(X_train, y_train, 'sigmoid', 6, 1.2, 4, 'scale', 0.5, True,
                                           True)  # works great for 0 values, doesn't work for 1 values

    y_pred = predict_y(svclassifier, X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'Class\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')


bankdata = pd.read_csv("data.csv")
print(bankdata.shape)  # number of lines and columns
print(bankdata.head())  # first 4 lines
"""Data processing -> divide data into attributes and labels
                      divide in training and data sets"""
attributes = bankdata.drop('Class', axis=1)  # will contain all columns except Class which is the label
labels = bankdata['Class']  # label
print(attributes)
print(labels)
# dividing data
X_train, X_test, y_train, y_test = train_test_split(attributes, labels,
                                                    test_size=0.10)  # 90% of data will be training, 10% test

# svcClassify1()  # basic default
svcClassify2()  # the precision is almost perfect for both 0 and 1
# svcClassify3()  #lower precision for both
# svcClassify4() #the accuracy is not the greatest
# svcClassify5()  #works great for 0 values, doesn't work for 1 values

"""
- macro avg = averaging the unweighted mean per label
- weighted average = averaging the support-weighted mean per label
- precision = truePositives / (truePositives + falsePositives)
- recall = truePositives / (truePositives + falseNegatives)
- support = number of occurrences of each class in y_test
"""
