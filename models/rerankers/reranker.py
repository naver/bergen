'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from abc import ABC, abstractmethod

class Reranker(ABC):
    def __init__(self, model_name=None):
        self.model_name = model_name

    @abstractmethod
    def __call__(self, kwargs):
        pass

    @abstractmethod
    def collate_fn(self, batch, query_or_doc=None):
        pass

