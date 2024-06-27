from tqdm import tqdm
from hydra.utils import instantiate
import torch

class GenerateQueries:
    def __init__(self, 
                 init_args=None,
                 ):
        # instaniate model
        self.model = instantiate(init_args)

    @torch.no_grad() 
    def eval(self, dataset):
        return self.model.generate(dataset["content"])

    def get_clean_model_name(self):
        return self.model.name
