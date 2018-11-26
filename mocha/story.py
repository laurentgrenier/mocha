from enum import Enum
import numpy as np
import platform
from datetime import datetime
from lib.utils import object_id
from lib.mongo import Mongo
from lib.logger import Logger


class Story:
    class Output(Enum):
        STD = 1
        CACHE = 2
        FILE = 3
        DATABASE = 4

    def __init__(self, name, persistent=True):
        self.name = name
        self.sessions = []
        self.session_id = ""
        self.epoch_id = 0
        self.persistent = persistent
        self.database = Mongo()
        self.log = Logger("story-{}".format(name))

    def session(self, session_id=None):
        if session_id:
            self.session_id = session_id

        return next(session for session in self.sessions if session["_id"] == self.session_id)

    def epoch(self, epoch_id=None, session_id=None):
        if epoch_id:
            self.epoch_id = epoch_id
        if session_id:
            self.session_id = session_id

        return next(epoch for epoch in self.session()["epochs"] if epoch["_id"] == self.epoch_id)

    def new_acc(self, value, epoch_id=None, session_id=None):
        if epoch_id:
            self.epoch_id = epoch_id
        if session_id:
            self.session_id = session_id

        self.epoch(self.epoch_id)["acc"] = np.append(self.epoch(self.epoch_id)["acc"], value)

        # save to the database
        if self.persistent:
            self.database.push(self.epoch_id, "acc", float(value), "epochs")

    def new_loss(self, value, epoch_id=None, session_id=None):
        if epoch_id:
            self.epoch_id = epoch_id
        if session_id:
            self.session_id = session_id

        self.epoch(self.epoch_id)["loss"] = np.append(self.epoch(self.epoch_id)["loss"], value)

        # save to the database
        if self.persistent:
            self.database.push(self.epoch_id, "loss", float(value), "epochs")

    def epoch_compute_acc(self, epoch_id=None, session_id=None):
        if epoch_id:
            self.epoch_id = epoch_id
        if session_id:
            self.session_id = session_id

        epoch_acc = np.mean(self.epoch(self.epoch_id)["acc"])
        self.epoch(self.epoch_id)["acc_mean"] = epoch_acc

        # save to the database
        if self.persistent:
            self.database.update(self.epoch_id, {"acc_mean": epoch_acc}, "epochs")

    def epoch_compute_loss(self, epoch_id=None, session_id=None):
        if epoch_id:
            self.epoch_id = epoch_id
        if session_id:
            self.session_id = session_id

        epoch_loss = np.mean(self.epoch(self.epoch_id)["loss"])
        self.epoch(self.epoch_id)["loss_mean"] = epoch_loss

        # save to the database
        if self.persistent:
            self.database.update(self.epoch_id, {"loss_mean": epoch_loss}, "epochs")

    def epoch_set(self, key, value, epoch_id=None, session_id=None):
        if epoch_id:
            self.epoch_id = epoch_id
        if session_id:
            self.session_id = session_id

        self.epoch(self.epoch_id)[key] = value

    def new_session(self, label):
        # create a session
        session_id = object_id()
        session = {"_id": session_id,
                   "time": datetime.now(),
                    "platform": platform.node(),
                    "label": label,
                    "epochs": [],
                    "acc": 0.0,
                    "loss": 0.0}

        # save to the database
        if self.persistent:
            self.database.upsert(session, "sessions")

        # add to the local sessions list
        self.sessions.append(session)

        # move the session cursor to the new session
        self.session_id = session_id

        # return the current session object
        return self.session(session_id)

    def new_epoch(self, num, session_id=None):
        if session_id:
            self.session_id = session_id
        self.epoch_id = object_id()

        # the new epoch
        epoch = {"_id": self.epoch_id,
                 "session": { "$ref": "epochs", "$id": self.session_id, "$db": "sessions"},
                 "rank": num, "acc": [], "loss": [], "acc_mean": 0.0, "loss_mean": 0.0}

        # save to the database
        self.database.upsert(epoch, "epochs")

        # add to the local epochs of the current session
        self.session()["epochs"].append(epoch)

        # move the epoch cursor to the new epoch
        return self.epoch(self.epoch_id)

    def close_epoch(self, epoch_id=None, session_id=None):
        if epoch_id:
            self.epoch_id = epoch_id
        if session_id:
            self.session_id = session_id

        self.epoch_compute_acc(self.epoch_id, self.session_id)
        self.epoch_compute_loss(self.epoch_id, self.session_id)

    def session_compute_acc(self, session_id=None):
        if session_id:
            self.session_id = session_id

        self.session()["acc"] = np.mean([epoch["acc_mean"] for epoch in self.session()["epochs"]])

        # save to the database
        if self.persistent:
            self.database.update(self.session_id, {"acc": self.session()["acc"]}, "sessions")

        return self.session()["acc"]

    def session_compute_loss(self, session_id=None):
        if session_id:
            self.session_id = session_id

        self.session()["loss"] = np.mean([epoch["loss_mean"] for epoch in self.session()["epochs"]])

        # save to the database
        if self.persistent:
            self.database.update(self.session_id, {"loss": self.session()["loss"]}, "sessions")

        return self.session()["loss"]

    def close_session(self, session_id=None):
        if session_id:
            self.session_id = session_id

        # accuracy and loss compute from epochs
        if len(self.session()["epochs"]) > 0:
            self.session_compute_acc()
            self.session_compute_loss()
