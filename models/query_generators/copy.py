from models.query_generators.query_generator import QueryGenerator
from typing import List

class CopyQuery(QueryGenerator):
    def __init__(self):
        self.name = "copy"
    
    def generate(self, user_questions: List[str]):
        return user_questions