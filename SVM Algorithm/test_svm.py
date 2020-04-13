import unittest
import pandas as pd
from sklearn import datasets
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


import SVMTest


class TestSVM(unittest.TestCase):

    def test_get_svclassifier(self):
        digits = datasets.load_digits()
        svclassifier = SVMTest.get_fitted_svclassifier(digits.data[:-1],digits.target[:-1], 'linear', 3, 1, 5, 'scale', 0.0, True, False)
        try:
            svclassifier.predict(digits.data[-1:])
        except NotFittedError as e:
            print('Model not fitted')


if __name__ == '__main__':
    unittest.main()
