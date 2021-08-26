import logging
from typing import Union

from bclm.models.api import Document

logger = logging.getLogger('bclm')


class Pipeline:

    def __init__(self):
        self.processors = []
        # set up processors
        for item in self.load_list:
            processor_name, _, _ = item
            logger.info('Loading: ' + processor_name)
            self.processors.append(load_processor(item))
        logger.info("Done loading processors!")

    def process(self, doc: Document):
        # run the pipeline
        for processor_name in PIPELINE_NAMES:
            if self.processors.get(processor_name):
                process = self.processors[processor_name].process
                doc = process(doc)
        return doc

    def __call__(self, doc):
        assert(any([isinstance(doc, str), isinstance(doc, list), isinstance(doc, Document)]),
               'input should be either str, list or Document')
        doc = self.process(doc)
        return doc
