from pymongo import MongoClient


class MongoDatabase:
    def __init__(self):
        self.__connection_string = None
        self.__database_name = None
        self.__client = None
        self.__db_handler = None
        self.__collection_handler = None

    def database_init(self, connection_string):
        self.__client = MongoClient(connection_string)

    def switch_database(self, database_name):
        self.__db_handler = self.__client[database_name]

    def switch_collection(self, collection_name):
        self.__collection_handler = self.__db_handler[collection_name]
        return self.__collection_handler
