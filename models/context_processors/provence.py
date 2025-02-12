from models.context_processors.context_processor import ContextProcessor
from transformers import AutoModel
from typing import List
import torch
from datasets.fingerprint import Hasher

class ProvenceCompressor(ContextProcessor):
    def __init__(
        self,
        model_name,
        name = "provence",
        **kwargs
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.kwargs = kwargs
        self.name = name+"_"+Hasher.hash(str(kwargs))

    def process(self, contexts: List[List[str]], queries: List[List[str]]):
        return self.model.process(queries, contexts, **self.kwargs)['pruned_context']