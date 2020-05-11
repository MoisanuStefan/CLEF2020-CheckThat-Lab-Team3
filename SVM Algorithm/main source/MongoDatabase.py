from pymongo import MongoClient


class MongoDatabase:
    def __init__(self, connection_string):
        self.__connection_string = connection_string
        self.__database_name = None
        self.__client = None
        self.__db_handler = None
        self.__collection_handler = None

    def database_init(self, database_name):
        self.__database_name = database_name
        self.__client = MongoClient(self.__connection_string)
        self.__db_handler = self.__client[self.__database_name]

    def set_collection(self, collection_name):
        self.__collection_handler = self.__db_handler[collection_name]
        return self.__collection_handler
