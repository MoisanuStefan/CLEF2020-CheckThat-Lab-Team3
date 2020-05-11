import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
from os import path
from MongoDatabase import MongoDatabase


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

    def predict(self, count):
        collection_handler = self.__database_handler.set_collection('tweetsFeatures')
        raw_predict_dataset = SVMAlgorithm.get_predict_dataset(collection_handler, count)
        predict_dataset = self.__tfidf_vectorizer.transform(raw_predict_dataset['tweet_text'].values)
        prediction_list = self.__model.predict(predict_dataset)
        collection_handler = self.__database_handler.set_collection('tweetsVerdict')
        object_id_list = raw_predict_dataset['_id'].values
        count = self.update_database_predictions(collection_handler, prediction_list, object_id_list)
        if count > 0:
            print('[LOG] Updated ' + str(count) + ' tweet predictions')

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
        return pd.DataFrame(collection_handler.find(None, {'tweet_text': 1}).sort('_id', -1).limit(count))

    @staticmethod
    def get_training_dataset(collection_handler):
        return pd.DataFrame(collection_handler.find(None, {'tweet_text': 1, 'verdict': 1})).drop('_id', axis=1)

    def process_train_data(self):
        collection_handler = self.__database_handler.set_collection('tweetsFeaturesTrainingModel')
        return SVMAlgorithm.get_training_dataset(collection_handler)

    def synchronize_predictions(self):
        collection_handler = self.__database_handler.set_collection('tweetsVerdict')
        cursor = collection_handler.aggregate([{
            '$lookup': {
                'from': 'tweetsFeatures',
                'localField': 'reference',
                'foreignField': '_id',
                'as': 'name'
            }
        }])
        tweets_text_dataset = []
        for x in cursor:
            if x['svm_verdict'] == -1:
                tweets_text_dataset.append({'tweet_id': x['reference'], 'tweet_text': x['name'][0]['tweet_text']})

        if len(tweets_text_dataset) > 0:
            pd_tweets_text_dataset = pd.DataFrame(tweets_text_dataset)
            predict_dataset = self.__tfidf_vectorizer.transform(pd_tweets_text_dataset['tweet_text'].values)
            prediction_list = self.__model.predict(predict_dataset)
            count = SVMAlgorithm.update_database_predictions(collection_handler, prediction_list,
                                                             pd_tweets_text_dataset['tweet_id'].values)
            print('[LOG] Synced ' + str(count) + ' tweet predictions')
        else:
            print('[LOG] No unsynced tweet predictions found, you\'re up to date')

    def init(self):
        self.database_connection()
        self.fit_model()
        self.synchronize_predictions()


# put this outside trigger function
obj = SVMAlgorithm()
obj.init()
# next: when signal is recived use obj.predict(count) in trigger function

# to do:
# - modify model serialization (load, save, model_already_exist)
