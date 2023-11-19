import logging
import logging.handlers
from typing import *
import pathlib


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def __init__(self):
        self.configured = False
        

    def getLogger(self, moduleName : str = "main"):
        if not self.configured:
            raise RuntimeError("Logger is not configured")

        logger = logging.getLogger(moduleName)

        if moduleName in self.configuredLoggers:
            return logger

        logger.propagate = False
        
        formatter = logging.Formatter(
        '%(asctime)s | %(name)s [%(threadName)s] |  %(levelname)s: %(message)s')
        logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if self.logFile is not None:
            file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.logFile.absolute(), when='midnight', backupCount=30)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        self.configuredLoggers.add(moduleName)

        return logger


    def configureLoggers(self, logFile : pathlib.Path | None = None) -> None:
        if self.configured:
            return

        self.appName = "HwAwareCutter"
        self.logFile = logFile
        self.configuredLoggers = set()

        self.configured = True