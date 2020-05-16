import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
from os import path
from MongoDatabase import MongoDatabase
from Database import update_start_time, update_end_time


class SVMAlgorithm:
    def __init__(self):
        self.__model = None
        self.__tfidf_vectorizer = None
        self.__database_handler = None

    def database_connection(self):
        self.__database_handler = MongoDatabase('mongodb+srv://watchdog:example@clef-uaic-svoxc.mongodb.net/'
                                                'test?retryWrites=true&w=majority')
        self.__database_handler.database_init('Tweets')

    @staticmethod
    def model_already_exist():
        return path.exists('model.joblib') and path.exists('tfidf_vectorizer.joblib')

    def load_model(self):
        self.__model = load('model.joblib')
        self.__tfidf_vectorizer = load('tfidf_vectorizer.joblib')

    def save_model(self):
        dump(self.__model, 'model.joblib')
        dump(self.__tfidf_vectorizer, 'tfidf_vectorizer.joblib')

    def train_svm(self, raw_training_dataset):
        self.__tfidf_vectorizer = TfidfVectorizer(binary=True, use_idf=True)
        training_dataset = self.__tfidf_vectorizer.fit_transform(raw_training_dataset['tweet_text'].values)
        training_verdict = raw_training_dataset['verdict'].values

        self.__model = SVC(probability=True, kernel='rbf')
        self.__model.fit(training_dataset, training_verdict)

    def fit_model(self):
        print('[LOG] Check if model exist...')
        if SVMAlgorithm.model_already_exist():
            print('[LOG] Model already exist and will be loaded...')
            self.load_model()
            print('[LOG] Model loaded successfully')
        else:
            print('[LOG] Model doesn\'t exist. I begin train procedure...')
            raw_train_dataset = self.process_train_data()
            self.train_svm(raw_train_dataset)
            self.save_model()
            print('[LOG] Model trained and saved successfully')

    @staticmethod
    def update_database_predictions(collection_handler, prediction_list, object_id_list):
        count = 0
        for index in range(0, len(prediction_list)):
            result = collection_handler.update_one({"reference": object_id_list[index]}, {"$set": {"svm_verdict": int(
                                                   prediction_list[index])}})
            count += result.modified_count
        return count

    @staticmethod
    def get_predict_dataset(collection_handler, count):
        tweet_text = collection_handler.find(None, {'full_text': 1}).sort('_id', -1).limit(count)
        actual_tweet_text = []
        for line in tweet_text:
            actual_tweet_text.append({'tweet_id': line['_id'], 'tweet_text': line['full_text']})
        return pd.DataFrame(actual_tweet_text)

    @staticmethod
    def get_training_dataset(collection_handler):
        return pd.DataFrame(collection_handler.find(None, {'tweet_text': 1, 'verdict': 1})).drop('_id', axis=1)

    def process_train_data(self):
        collection_handler = self.__database_handler.set_collection('tweetsFeaturesTrainingModel')
        return SVMAlgorithm.get_training_dataset(collection_handler)

    def synchronize_predictions(self):
        # conectare la baza de date
        self.database_connection()
        collection_handler = self.__database_handler.set_collection('tweetsVerdict_v1')
        join_cursor = collection_handler.aggregate([{
            '$match': {'svm_verdict': -1}
        }, {
            '$lookup': {
                'from': 'filteredTweets_v1',
                'localField': 'reference',
                'foreignField': '_id',
                'as': 'join'
            }
        }
        ])
        tweets_text_dataset = []
        for line in join_cursor:
            tweets_text_dataset.append({'tweet_id': line['reference'], 'tweet_text': line['join'][0]['full_text']})
        if len(tweets_text_dataset) > 0:
            # posibila modificare collection_handler pentru load din baza de date
            self.fit_model()
            pd_tweets_text_dataset = pd.DataFrame(tweets_text_dataset)
            predict_dataset = self.__tfidf_vectorizer.transform(pd_tweets_text_dataset['tweet_text'].values)
            prediction_list = self.__model.predict(predict_dataset)
            count = SVMAlgorithm.update_database_predictions(collection_handler, prediction_list,
                                                             pd_tweets_text_dataset['tweet_id'].values)
            print('[LOG] Synced ' + str(count) + ' tweet predictions')
        else:
            print('[LOG] No unsynced tweet predictions found, you\'re up to date')

    def predict(self, count):
        update_start_time('svm')
        self.database_connection()
        # posibila modificare collection_handler pentru load din baza de date
        self.fit_model()
        collection_handler = self.__database_handler.set_collection('filteredTweets_v1')
        raw_predict_dataset = SVMAlgorithm.get_predict_dataset(collection_handler, count)
        predict_dataset = self.__tfidf_vectorizer.transform(raw_predict_dataset['tweet_text'].values)
        prediction_list = self.__model.predict(predict_dataset)
        collection_handler = self.__database_handler.set_collection('tweetsVerdict_v1')
        object_id_list = raw_predict_dataset['tweet_id'].values
        count = self.update_database_predictions(collection_handler, prediction_list, object_id_list)
        if count > 0:
            print('[LOG] Updated ' + str(count) + ' tweet predictions')
        else:
            print('[LOG] No unpredicted tweets found!')
        update_end_time('svm')


obj = SVMAlgorithm()
# obj.synchronize_predictions()
obj.predict(5)
