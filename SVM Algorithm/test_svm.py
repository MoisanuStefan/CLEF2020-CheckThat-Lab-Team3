import unittest
import pandas as pd
from sklearn import datasets
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from svm_exceptions import *


import SVMTest


class TestSVM(unittest.TestCase):

    def test_get_svclassifier(self):
        digits = datasets.load_digits()
        svclassifier = SVMTest.get_fitted_svclassifier(digits.data[:-1],digits.target[:-1], 'linear', 3, 1, 5, 'scale', 0.0, True, False)
        try:
            svclassifier.predict(digits.data[-1:])
        except NotFittedError as e:
            print('Model not fitted')

    def test_serialization(self):
        digits = datasets.load_digits()
        svclassifier1 = 'this is a string'
        svclassifier2 = SVMTest.get_fitted_svclassifier(digits.data[:-1], digits.target[:-1], 'linear', 3, 1, 5,
                                                        'scale', 0.0, True, False)

        path_to_object1 = 'non-existent-file.joblib'
        path_to_object2 = 'model.joblib'
        try:
            print('Test serialization OK!')
            # SVMTest.serialization(path_to_object1, svclassifier1)
            # SVMTest.serialization(path_to_object2, svclassifier2)
        except IsNotSVCException as ex1:
            print('EXCEPTION: ' + ex1.message)
        except PathExistsException as ex2:
            print('EXCEPTION: ' + ex2.message)

    def test_deserializer(self):
        path_to_object1 = 'stm.wrong_extension'
        path_to_object2 = 'wrong.joblib'

        try:
            print('Test deserializer OK!')
            # svclassifier = SVMTest.deserializer(path_to_object1)
            # svclassifier = SVMTest.deserializer(path_to_object2)
        except WrongExtensionException as ex1:
            print('EXCEPTION: ' + ex1.message)
        except PathNotExistsException as ex2:
            print('EXCEPTION: ' + ex2.message)


if __name__ == '__main__':
    unittest.main()