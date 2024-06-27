from abc import ABC, abstractmethod
from typing import List

class QueryGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, user_questions: List[str]):
        pass