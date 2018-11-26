from pymongo import MongoClient
from lib.utils import object_id
from lib.logger import Logger
import time
from bson.objectid import ObjectId

class Mongo:
    def __init__(self, host="localhost", port=27017, database="draft", collection="sessions"):
        self.host = host
        self.port = port
        self.database_name = database
        self.collection = collection
        self.client = None
        self.database = None
        self.log = Logger("mongodb")

    def __connect(self):
        self.client = MongoClient('localhost', 27017)
        self.database = self.client[self.database_name]

    def __disconnect(self):
        self.client.close()

    def upsert(self, data, collection=None):
        if not "_id" in data:
            data["_id"] = object_id()
        if not collection:
            collection = self.collection

        self.__connect()
        collection = self.database[collection]
        collection.update_one({"_id": data["_id"]}, {"$set": data}, upsert=True)
        self.__disconnect()

        return data

    def push(self, id, field, data, collection=None):
        if not collection:
            collection = self.collection
        self.log.write("push _id:{}, {}:{}".format(id, field, data))
        self.__connect()
        collection = self.database[collection]
        collection.update_one({"_id": id}, {"$push": {field: data}})
        self.__disconnect()

        return data

    def post(self, data, collection):
        self.__connect()
        collection = self.database[collection]
        inserted_id = collection.insert_one(data).inserted_id
        self.__disconnect()

        data["_id"] = inserted_id
        return data

    def update(self, id, data, collection):
        self.log.write("updating _id:{}, {} into collection {}".format(id, data, collection))
        self.__connect()
        collection = self.database[collection]
        collection.update_one({"_id": id}, {"$set": data}, upsert=False)
        self.__disconnect()

        return data

    def find_one(self, filter={}, collection="others"):
        self.__connect()
        collection = self.database[collection]
        result = collection.find_one(filter)
        self.__disconnect()

        return result

    def find_one_by_id(self, id, collection="others"):
        self.__connect()
        collection = self.database[collection]
        result = collection.find_one({"_id": ObjectId(id)})
        self.__disconnect()

        return result

    def find(self, filter={}, collection="others"):
        self.__connect()
        collection = self.database[collection]
        result = collection.find(filter)
        self.__disconnect()

        return result

    def all(self, collection="others"):
        self.__connect()
        collection = self.database[collection]
        result = collection.find()
        self.__disconnect()

        return result

    def watcher(self, collection="others"):
        self.__connect()
        collection = self.database[collection]

        cursor = collection.find(tailable=True)
        while cursor.alive:
            try:
                doc = cursor.next()
                print("watcher test: ", doc)
            except StopIteration:
                time.sleep(1)