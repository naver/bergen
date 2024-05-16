'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from models.retrievers.retriever import Retriever

class OracleProvenance(Retriever): 

    def __init__(self, model_name=None):
        self.model_name = model_name      

    def collate_fn(self, *args, **kwargs):
        pass

    def similarity_fn(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


