from sklearn.feature_extraction.text import TfidfVectorizer
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
        return self.__best

    def get_statistics(self):
        print('[LOG] C: ' + str(self.__best.C) + ' | gamma: '
              + str(self.__best.gamma) + ' | kernel: ' + str(self.__best.kernel))

    def search(self, collection_handler):
        raw_dataset = GridSearch.extract_datasets()
        train_dataset = GridSearch.normalize(raw_dataset)
        best_model = self.grid_fit(raw_dataset, train_dataset)
        collection_handler.update_one({'file_name': 'best_svc_parameters'},
                                      {'$set': {'C': best_model.C,
                                                'kernel': best_model.kernel, 'gamma': best_model.gamma}}, upsert=True)

