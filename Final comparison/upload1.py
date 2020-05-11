import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from pymongo import MongoClient



def svcClassify2():
    svclassifier = SVC(kernel='poly',degree=4,C=1.5,random_state=3,gamma='auto',coef0=0.1)

    svclassifier.fit(X_train, y_train)  # actual training

    y_pred = svclassifier.predict(X_test)
    matrix = classification_report(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('These are the real values of \'verdict\' column for the test set:')
    for i in y_test:
        print(i, end=' ')
    print('\nThese are the predicted values by the svm module:')
    for i in y_pred:
        print(i, end=' ')

    cluster = MongoClient("mongodb+srv://user1:password_1@cluster0-izgs7.mongodb.net/test?retryWrites=true&w=majority")
    db = cluster["test"]
    collection = db["tweets"]
    k=0;
    for i in y_pred:
        if k < 5 :
            post = {"id2": k, "verdict": i}
            k = k+1
            collection.insert_one(post)


bankdata = pd.read_csv("tweets1.csv")
print(bankdata.shape)  # number of lines and columns
print(bankdata.head())  # first 4 lines
"""Data processing -> divide data into attributes and labels
                      divide in training and data sets"""
attributes = bankdata.drop('tweet_text', axis=1)  # will contain all columns except Class which is the label
labels = bankdata['verdict']  # label
print(attributes)
print(labels)
# dividing data
X_train, X_test, y_train, y_test = train_test_split(attributes, labels, test_size=0.10)  # 90% of data will be training, 10% test

svcClassify2()

