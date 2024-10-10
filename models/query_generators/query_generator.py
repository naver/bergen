from abc import ABC, abstractmethod
from typing import List
from models.generators.generator import Generator

class QueryGenerator(ABC):
    generator = None
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, user_questions: List[str]):
        pass

    def init_generator(self, generator: Generator):
        self.generator = generator