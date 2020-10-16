import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from MongoDatabase import MongoDatabase
from Database import update_start_time, update_end_time
import pickle
from GridSearch import GridSearch


class SVMAlgorithm:
    def __init__(self):
        self.__model = None
        self.__tfidf_vectorizer = None
        self.__client_handler = None

    def database_connection(self):
        self.__client_handler = MongoDatabase()
        self.__client_handler.database_init('mongodb+srv://watchdog:example@clef-uaic-svoxc.mongodb.net/'
                                            'test?retryWrites=true&w=majority')

    def database_switcher(self, database_name=None, collection_name=None):
        if database_name is not None:
            self.__client_handler.switch_database(database_name)
        if collection_name is not None:
            return self.__client_handler.switch_collection(collection_name)

    @staticmethod
    def model_already_exist(modelname, collection_handler):
        if modelname == 'svm_model' or modelname == 'tfidf_vectorizer':
            return True if collection_handler.count_documents({'file_name': modelname}, limit=1) == 1 else False
        return False

    def load_model(self, modelname, collection_handler):
        if modelname == 'svm_model':
            self.__model = pickle.loads(collection_handler.find_one({'file_name': modelname},
                                                                    {'content': 1, '_id': 0})['content'])
        elif modelname == 'tfidf_vectorizer':
            self.__tfidf_vectorizer = pickle.loads(collection_handler.find_one({'file_name': modelname},
                                                                               {'content': 1, '_id': 0})['content'])
        else:
            print('[LOG] Wrong model name passed for LOAD')

    def save_model(self, modelname, collection_handler):
        if modelname == 'svm_model':
            bin_model_file = pickle.dumps(self.__model)
            collection_handler.update_one({'file_name': modelname}, {'$set': {'content': bin_model_file}},
                                          upsert=True)
        elif modelname == 'tfidf_vectorizer':
            bin_tfidf_file = pickle.dumps(self.__tfidf_vectorizer)
            collection_handler.update_one({'file_name': modelname}, {'$set': {'content': bin_tfidf_file}},
                                          upsert=True)
        else:
            print('[LOG] Wrong model name passed for SAVE')

    def train_tfidf(self, raw_training_dataset):
        self.__tfidf_vectorizer = TfidfVectorizer(binary=True, use_idf=True)
        training_dataset = self.__tfidf_vectorizer.fit_transform(raw_training_dataset['tweet_text'].values)
        training_verdict = raw_training_dataset['verdict'].values
        return training_dataset, training_verdict

    def train_svm(self, collection_handler, training_dataset, training_verdict):
        svc_parameters = collection_handler.find_one({'file_name': 'best_svc_parameters'},
                                                     {'C': 1, 'gamma': 1, 'kernel': 1, '_id': 0})
        self.__model = SVC(C=svc_parameters['C'], kernel=svc_parameters['kernel'], gamma=svc_parameters['gamma'])
        self.__model.fit(training_dataset, training_verdict)

    @staticmethod
    def new_train_dataset(collection_handler):
        return collection_handler.find_one({'file_name': 'gridsearch_check'},
                                           {'traindata_updated': 1, '_id': 0})['traindata_updated']

    def fit_model(self):
        raw_train_dataset = self.process_train_data()

        collection_handler = self.database_switcher('Files', 'svmData')

        if not SVMAlgorithm.new_train_dataset(collection_handler):
            print('[LOG] New training dataset detected, I\'m looking for best parameters...')
            param_grid = collection_handler.find_one({'file_name': 'possible_svc_parameters'},
                                                     {'C': 1, 'gamma': 1, 'kernel': 1, '_id': 0})
            grid_search = GridSearch(param_grid)
            grid_search.search(collection_handler)
            grid_search.get_statistics()
            collection_handler.update_one({'file_name': 'gridsearch_check'}, {"$set": {'traindata_updated': True}})
            print('[LOG] Best parameters found and saved in database')

        print('[LOG] Check if TFIDF Vectorizer exists...')
        if SVMAlgorithm.model_already_exist('tfidf_vectorizer', collection_handler):
            print('[LOG] TFIDF Vectorizer already exists and will be loaded...')
            self.load_model('tfidf_vectorizer', collection_handler)
            processed_dataset = self.train_tfidf(raw_train_dataset)
            print('[LOG] TFIDF Vectorizer loaded successfully')
        else:
            print('[LOG] TFIDF Vectorizer doesn\'t exist. I begin train procedure...')
            processed_dataset = self.train_tfidf(raw_train_dataset)
            self.save_model('tfidf_vectorizer', collection_handler)
            print('[LOG] TFIDF Vectorizer trained and saved successfully')

        print('[LOG] Check if SVM Model exists...')
        if SVMAlgorithm.model_already_exist('svm_model', collection_handler):
            print('[LOG] SVM Model already exists and will be loaded...')
            self.load_model('svm_model', collection_handler)
            print('[LOG] SVM Model loaded successfully')
        else:
            print('[LOG] SVM Model doesn\'t exist. I begin train procedure...')
            self.train_svm(collection_handler, processed_dataset[0], processed_dataset[1])
            self.save_model('svm_model', collection_handler)
            print('[LOG] SVM Model trained and saved successfully')

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
        collection_handler = self.database_switcher('Tweets', 'tweetsFeaturesTrainingModel')
        return SVMAlgorithm.get_training_dataset(collection_handler)

    def synchronize_predictions(self):
        self.database_connection()
        collection_handler = self.database_switcher('Tweets', 'tweetsVerdict_v1')

        collection_handler.update_many({}, {"$set": {"svm_verdict": -1}})

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
        # update_start_time('svm')
        self.database_connection()
        self.fit_model()
        collection_handler = self.database_switcher('Tweets', 'filteredTweets_v1')
        raw_predict_dataset = SVMAlgorithm.get_predict_dataset(collection_handler, count)
        predict_dataset = self.__tfidf_vectorizer.transform(raw_predict_dataset['tweet_text'].values)
        prediction_list = self.__model.predict(predict_dataset)
        collection_handler = self.database_switcher('Tweets', 'tweetsVerdict_v1')
        object_id_list = raw_predict_dataset['tweet_id'].values
        count = self.update_database_predictions(collection_handler, prediction_list, object_id_list)
        if count > 0:
            print('[LOG] Updated ' + str(count) + ' tweet predictions')
        else:
            print('[LOG] No unpredicted tweets found!')
        # update_end_time('svm')


obj = SVMAlgorithm()
# obj.synchronize_predictions()
obj.predict(3)
