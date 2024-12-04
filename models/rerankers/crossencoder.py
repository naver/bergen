'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


from models.rerankers.reranker import Reranker
from peft import PeftModel, PeftConfig

class CrossEncoder(Reranker):
    def __init__(self, model_name=None,tokenizer_name=None, max_len=512):
        self.model_name = model_name
        self.max_len= max_len
        if "llama" in model_name:
            config = PeftConfig.from_pretrained(model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16,num_labels=1)
            self.model = PeftModel.from_pretrained(base_model, model_name)
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,num_labels=1)
            
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_length=self.max_len, padding_side="right")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_len)
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
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
        print(self.tokenizer)
        score = self.model(**kwargs.to('cuda')).logits
        return {
                "score": score
            }