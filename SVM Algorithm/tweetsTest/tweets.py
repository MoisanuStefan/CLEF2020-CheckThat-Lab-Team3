import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def svcClassify1():
    svclassifier = SVC(kernel='linear', degree=3, C=1, random_state=5)  # basic default

    svclassifier.fit(X_train, y_train)  # actual training

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'Class\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')


def svcClassify2():
    svclassifier = SVC(kernel='poly',degree=4,C=1.5,random_state=3,gamma='auto',coef0=0.1)

    svclassifier.fit(X_train, y_train)  # actual training

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'verdict\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')



def svcClassify3():
    svclassifier = SVC(kernel='poly',degree=4,C=1.5,random_state=3,gamma='auto',coef0=0.1,shrinking=False) #lower precision for both
    svclassifier.fit(X_train, y_train)  # actual training

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'Class\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')



def svcClassify4():
    svclassifier = SVC(kernel='linear',degree=1,C=1.7,random_state=2,gamma='scale',coef0=0.06,verbose=False) #the accuracy is not the greatest
    svclassifier.fit(X_train, y_train)  # actual training

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'Class\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')



def svcClassify5():
    svclassifier = SVC(kernel='sigmoid',degree=6,C=1.2,random_state=4,gamma='scale',coef0=0.5,verbose=True) #works great for 0 values, doesn't work for 1 values

    svclassifier.fit(X_train, y_train)  # actual training

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'verdict\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')





def svcClassify6():
    svclassifier = SVC(gamma=10)

    svclassifier.fit(X_train, y_train)  # actual training

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'Class\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')




bankdata = pd.read_csv("tweetsCSV.csv")
print(bankdata.shape)  # number of lines and columns
print(bankdata.head())  # first 4 lines
"""Data processing -> divide data into attributes and labels
                      divide in training and data sets"""
attributes = bankdata.drop('tweet_text', axis=1)  # will contain all columns except Class which is the label
labels = bankdata['verdict']  # label
print(attributes)
print('aici')
print(labels)
# dividing data
X_train, X_test, y_train, y_test = train_test_split(attributes, labels, test_size=0.10)  # 90% of data will be training, 10% test

#C, regularization parameter: double, the bigger it is the bigger the mistakes are
#


#svcClassify1() #basic default
svcClassify2()
#svcClassify3()  #lower precision for both
#svcClassify4() #the accuracy is not the greatest
#svcClassify5()  #works great for 0 values, doesn't work for 1 values
