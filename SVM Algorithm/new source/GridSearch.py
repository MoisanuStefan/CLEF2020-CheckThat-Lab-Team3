from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from MongoDatabase import MongoDatabase
import pandas as pd


class GridSearch:
    def __init__(self, param_grid):
        self.__grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, n_jobs=-1, scoring='average_precision')
        self.__best = None

    @staticmethod
    def extract_datasets():
        client_handler = MongoDatabase()
        client_handler.database_init('mongodb+srv://watchdog:example@clef-uaic-svoxc.mongodb.net/'
                                     'test?retryWrites=true&w=majority')
        client_handler.switch_database('Tweets')
        collection_handler = client_handler.switch_collection('tweetsFeaturesTrainingModel')

        tweet_text = collection_handler.find(None, {'tweet_text': 1, 'verdict': 1, '_id': 0})
        actual_tweet_text = []
        for line in tweet_text:
            actual_tweet_text.append({'tweet_text': line['tweet_text'], 'verdict': line['verdict']})
        text_for_train = pd.DataFrame(actual_tweet_text)
        return text_for_train

    @staticmethod
    def normalize(text_for_train):
        tfidf_vectorizer = TfidfVectorizer(binary=True, use_idf=True)
        training_dataset = tfidf_vectorizer.fit_transform(text_for_train['tweet_text'].values)
        return training_dataset

    def grid_fit(self, text_for_train, training_dataset):
        self.__grid.fit(training_dataset, text_for_train['verdict'].values)
        self.__best = self.__grid.best_estimator_
        print(self.__best)

    def get_statistics(self):
        print('Some details about precisiton')

    def search(self):
        raw_dataset = GridSearch.extract_datasets()
        train_dataset = GridSearch.normalize(raw_dataset)
        self.grid_fit(raw_dataset, train_dataset)


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
obj = GridSearch(param_grid)
obj.search()
