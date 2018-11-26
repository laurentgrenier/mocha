from enum import Enum
import uuid
from datetime import datetime
import os

class Logger:
    def __init__(self, name):
        self.name = "model_{}_{}_{}.log".format(name, str(uuid.uuid4()), datetime.now())
        self.min_rank = 0

    class Output(Enum):
        STD = 1
        CACHE = 2
        FILE = 3
        DATABASE = 4

    class Level(Enum):
        DEBUG = {"display": "(o.o)", "rank": 0}
        INFO = {"display": "(o.-)", "rank": 1}
        WARNING = {"display": "(o.O)", "rank": 2}
        ERROR = {"display": "(x.x)", "rank": 3}
        SUCCESS = {"display": "(^.^)", "rank": 4}

    def active(self, level):
        if "min_logger_rank" in os.environ:
            self.min_rank = int(os.environ['min_logger_rank'])

        return level.value["rank"] >= self.min_rank

    def write(self, message, level=Level.INFO, indent=0, output=Output.STD):
        if self.active(level):
            if output == self.Output.STD:
                print("{} {} {}".format("\t" * indent, level.value["display"], message))
