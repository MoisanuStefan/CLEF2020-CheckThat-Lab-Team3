import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import unittest


def svcClassify11(kern,degr,randomState,gama,verbos):
    svclassifier = SVC(kernel=kern,degree=degr,random_state=randomState,gamma=gama,verbose=verbos) #the accuracy is not the greatest
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
X_train, X_test, y_train, y_test = train_test_split(attributes, labels, test_size=0.10)  # 90% of data will be training, 10% test

#C, regularization parameter: double, the bigger it is the bigger the mistakes are
#


#svcClassify1() #basic default
#svcClassify2()
#svcClassify3()  #lower precision for both
#svcClassify4() #the accuracy is not the greatest
#svcClassify5()  #works great for 0 values, doesn't work for 1 values
#svcClassify11('linear',3,2,'auto',False)

kern='linear'
degr=3
randomState=2
gama='auttto'
verbs=False



class SimpleTest(unittest.TestCase):
        def test_parameters(self):
             if kern != 'linear' and kern != 'poly' and kern!= 'rbf' and kern!= 'sigmoid':
                 print(kern+' is not a valid value for kernel parameter')
             elif degr < 0:
                print( str(degr)+' is not a valid value for degree parameter')
             elif randomState <0:
                print(str(randomState) + ' is not a valid value for random_state parameter')
             elif gama != 'scale' and gama != 'auto' :
                print(gama + ' is not a valid value for gamma parameter')
             elif verbs != False and verbs!=True :
                print(verbs + ' is not a valid value for verbose parameter')
             else:
                svcClassify11(kern,degr,randomState,gama,verbs)

if __name__ == '__main__':
    unittest.main()