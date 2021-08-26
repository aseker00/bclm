from bclm.models.api import Document
from bclm.pipeline.processor import Processor


class MWTProcessor(Processor):

    def __init__(self):
        super().__init__()
        self._md = MD(model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, doc: Document):
        pass
