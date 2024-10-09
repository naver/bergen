from models.query_generators.query_generator import QueryGenerator
from typing import List
from tqdm import tqdm
from datasets import Dataset

class UnfoldAPIQuery(QueryGenerator):
    def __init__(self, name="unfolded_query", model="generator", prompt=""):
        self.name = name
        self.model = model
        self.prompt = prompt
        
    def generate(self, user_questions: List[str]):
        retriever_queries = []
        dataset = Dataset.from_dict({'query': [self.prompt.format(user_prompt=q) for q in user_questions], 'q_id': list(range(len(user_questions))), 'label': ["" for _ in range(len(user_questions))]})
        query_ids, _, _, retriever_queries, _, _ = self.generator.eval(dataset)
        retriever_queries = [x for _, x in sorted(zip(query_ids, retriever_queries), key=lambda pair: pair[0])]
        return retriever_queries