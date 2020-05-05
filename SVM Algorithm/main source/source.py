import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
from os import path


class SVMAlgorithm:
    def __init__(self):
        self.model = None
        self.tfidf_vectorizer = None

    def request_dataset_from_database(self, database_info, table_name):
        dataset = pd.read_csv(table_name)
        return dataset[['tweet_text', 'verdict']]

    def get_training_dataset(self, database_info):
        return self.request_dataset_from_database(database_info, 'training_data.csv')

    def get_test_dataset(self, database_info):
        return self.request_dataset_from_database(database_info, 'test_data.csv')

    def train_svm(self, raw_training_dataset):
        if path.exists('model.joblib') and path.exists('tfidf_vectorizer.joblib'):
            self.model = load('model.joblib')
            self.tfidf_vectorizer = load('tfidf_vectorizer.joblib')
        else:
            self.tfidf_vectorizer = TfidfVectorizer(binary=True, use_idf=True)
            training_dataset = self.tfidf_vectorizer.fit_transform(raw_training_dataset['tweet_text'].values)
            training_verdict = raw_training_dataset['verdict'].values

            self.model = SVC(probability=True, kernel='rbf')
            self.model.fit(training_dataset, training_verdict)
            dump(self.model, 'model.joblib')
            dump(self.tfidf_vectorizer, 'tfidf_vectorizer.joblib')

    def predict(self, preprocessed_input_data):
        return self.model.predict(preprocessed_input_data).tolist()

    def wait_for_input(self):
        print('Block state function that waits for predictions request from other microservices')

    def test(self, precentage):
        raw_dataset = self.request_dataset_from_database('#database_connection_info', 'dataset.csv')

        length = len(raw_dataset)
        pos = int(length * precentage)
        raw_train_data = raw_dataset[:pos]
        raw_test_data = raw_dataset[pos:]

        self.train_svm(raw_train_data)

        test_dataset = self.tfidf_vectorizer.transform(raw_test_data['tweet_text'].values)
        test_verdict = raw_test_data['verdict'].values

        predictions = self.model.predict(test_dataset)
        print(confusion_matrix(test_verdict, predictions))
        print(classification_report(test_verdict, predictions))

    def main(self):
        dataset = self.get_training_dataset('#database_connection_info')
        self.train_svm(dataset)
        self.wait_for_input()


obj = SVMAlgorithm()
obj.test(0.8)
