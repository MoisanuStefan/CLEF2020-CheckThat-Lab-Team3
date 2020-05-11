import pymongo
from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://user1:password_1@cluster0-izgs7.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["test"]
collection = db["tweets"]


for i in range(5):
    print("Tweet number: " + str(i))
    results = collection.find({"id2": i})
    results2 = collection.find({"_id": i})
    for result in results:
        print("Result method no one:" + result["verdict"])
        for result2 in results2:
            print("Result method no 2: " + result2["verdict"])
            if result["verdict"] == result2["verdict"] == "1":
                print("The tweet is true")
            elif result["verdict"] == result2["verdict"] == "0":
                print("The tweet is fake")
            else:
                print("We are only 50% sure the tweet is true")