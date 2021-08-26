from abc import ABC, abstractmethod

from bclm.models.api import Document


class Processor(ABC):

    def __init__(self, config, pipeline):
        self._config = config
        self._pipeline = pipeline

    @abstractmethod
    def process(self, doc: Document):
        pass

    @property
    def config(self):
        return self._config

    @property
    def pipeline(self):
        return self._pipeline
