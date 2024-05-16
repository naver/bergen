'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


from models.rerankers.reranker import Reranker

class CrossEncoder(Reranker):
    def __init__(self, model_name=None,max_len=512):
        self.model_name = model_name
        self.max_len= max_len
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_len)
        self.model.eval()

    def collate_fn(self, examples):
        question = [e['query'] for e in examples]
        doc = [e['doc'] for e in examples]
        q_id = [e['q_id'] for e in examples]
        d_id = [e['d_id'] for e in examples]
        inp_dict = self.tokenizer(question, doc, padding="max_length", truncation='only_second', max_length=self.max_len,return_tensors='pt')
        inp_dict['q_id'] = q_id
        inp_dict['d_id'] = d_id
        return inp_dict

    def __call__(self, kwargs):
        score = self.model(**kwargs.to('cuda')).logits
        return {
                "score": score
            }