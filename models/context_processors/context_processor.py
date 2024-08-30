from abc import ABC, abstractmethod
from typing import List

class ContextProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, contexts: List[List[str]], queries: List[List[str]]):
        pass