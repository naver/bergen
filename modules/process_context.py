from tqdm import tqdm
from hydra.utils import instantiate
import torch
from typing import List

class ProcessContext:
    def __init__(self, 
                 init_args=None,
                 ):
        # instaniate model
        self.model = instantiate(init_args)

    @torch.no_grad() 
    def eval(self, contexts: List[List[str]], queries: List[List[str]]):
        return self.model.process(contexts, queries)

    def get_clean_model_name(self):
        return self.model.name
