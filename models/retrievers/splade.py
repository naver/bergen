'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from models.retrievers.retriever import Retriever


class Splade(Retriever):
    def __init__(self, model_name, max_len=512, query_encoder_name=None):

        self.model_name = model_name
        self.max_len = max_len
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        if query_encoder_name:
            self.query_encoder = AutoModelForMaskedLM.from_pretrained(query_encoder_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            self.query_encoder = self.model  # otherwise symmetric
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_len)
        self.reverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        if query_encoder_name:
            self.query_encoder = self.query_encoder.to(self.device)
            self.query_encoder.eval()
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            if query_encoder_name:
                self.query_encoder = torch.nn.DataParallel(self.query_encoder)
        
    def __call__(self, query_or_doc, kwargs):
        kwargs = {key: value.to('cuda') for key, value in kwargs.items()}
        if query_or_doc == "doc":
            outputs = self.model(**kwargs)
        else:  # query_or_doc == "query"
            outputs = self.query_encoder(**kwargs)
        outputs = self.model(**kwargs).logits

        # pooling over hidden representations
        emb, _ = torch.max(torch.log(1 + torch.relu(outputs)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)

        return {
                "embedding": emb
            }

    def collate_fn(self, batch, query_or_doc=None):
        key = 'generated_query' if query_or_doc=="query" else "content"
        content = [sample[key] for sample in batch]
        return_dict = self.tokenizer(content, padding=True, truncation=True, max_length= self.max_len, return_tensors='pt')
        return return_dict

    def similarity_fn(self, query_embds, doc_embds):
        return torch.sparse.mm(query_embds.to_sparse(), doc_embds.t()).to_dense()
        #return torch.mm(query_embds, doc_embds.t())