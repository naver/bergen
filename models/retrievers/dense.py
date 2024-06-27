'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoModel, AutoTokenizer
import torch
from models.retrievers.retriever import Retriever

class Dense(Retriever):
    #low_cpu_mem_usage=True,

    def __init__(self, model_name, max_len, pooler, similarity, prompt_q=None, prompt_d=None, query_encoder_name=None):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float16, trust_remote_code=True)
        if query_encoder_name:
            self.query_encoder = AutoModel.from_pretrained(query_encoder_name, torch_dtype=torch.float16, trust_remote_code=True)
        else:
            self.query_encoder = self.model  # otherwise symmetric
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        if query_encoder_name:
            self.query_encoder = self.query_encoder.to(self.device)
            self.query_encoder.eval()
        self.max_len = max_len
        self.similarity = similarity
        self.pooler = pooler
        self.prompt_q = "" if prompt_q is None else prompt_q
        self.prompt_d = "" if prompt_d is None else prompt_d 
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            if query_encoder_name:
                self.query_encoder = torch.nn.DataParallel(self.query_encoder)

    def __call__(self, query_or_doc, kwargs):
        kwargs = {key: value.to(self.device) for key, value in kwargs.items()}
        if query_or_doc == "doc":
            outputs = self.model(**kwargs)
        else:  # query_or_doc == "query"
            outputs = self.query_encoder(**kwargs)
        # pooling over hidden representations
        emb = self.pooler.pool(outputs[0], kwargs['attention_mask'])
        return {
                "embedding": emb
            }

    def collate_fn(self, batch, query_or_doc=None):
        key = 'generated_query' if query_or_doc=="query" else "content"
        content = [sample[key] for sample in batch]
        # some retrieval models have a "prompt" prefix, e.g. intfloat/e5-large-v2
        if query_or_doc == "query":
            content = ["{}{}".format(self.prompt_q, text) for text in content]
        if query_or_doc == "doc":
            content = ["{}{}".format(self.prompt_d, text) for text in content]
        return_dict = self.tokenizer(content, padding="longest", truncation="longest_first", max_length=self.max_len, return_tensors='pt')
        return return_dict
    

    def similarity_fn(self, query_embds, doc_embds):
        return self.similarity.sim(query_embds, doc_embds)

class MeanPooler:

    @staticmethod
    def pool(outputs, mask):
        outputs = outputs.masked_fill(~mask[..., None].bool(), 0.)
        return outputs.sum(dim=1) / mask.sum(dim=1)[..., None]

class ClsPooler:

    @staticmethod
    def pool(outputs, *args):
        return outputs[:,0]
    
class DotProduct:

    @staticmethod
    def sim(query_embds, doc_embds):
        return torch.mm(query_embds, doc_embds.t())

class CosineSim:

    @staticmethod
    def sim(query_embds, doc_embds):
        query_embds = query_embds / (torch.norm(query_embds, dim=-1, keepdim=True) + 1e-9)
        doc_embds = doc_embds / (torch.norm(doc_embds, dim=-1, keepdim=True) + 1e-9)
        return torch.mm(query_embds, doc_embds.t())
