'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import AutoPeftModelForCausalLM, PeftConfig


from models.rerankers.reranker import Reranker

class LLMCrossEncoder(Reranker):
    def __init__(self, model_name=None,max_len=2048, prompt=None):
        self.model_name = model_name
        self.max_len = max_len
        self.prompt = prompt
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_len, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.token_false_id = self.tokenizer.get_vocab()["false"]
        self.token_true_id = self.tokenizer.get_vocab()["true"]
        
        
    def create_instruction(self,sample):
        query = sample['query']
        doc = sample['doc']
        return eval(self.prompt)
    
    def collate_fn(self, examples):
        #query = [e['query'] for e in examples]
        #doc = [e['doc'] for e in examples]
        q_id = [e['q_id'] for e in examples]
        d_id = [e['d_id'] for e in examples]
        instr = [self.create_instruction(sample) for sample in examples]  # Add prompt to each text
        inp_dict = self.tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
        inp_dict['q_id'] = q_id
        inp_dict['d_id'] = d_id
        return inp_dict

    def __call__(self, kwargs):
        
        logits = self.model(**kwargs.to('cuda')).logits 
        model_scores = logits[:, -1, [self.token_false_id, self.token_true_id]].float()
        scores = torch.softmax(model_scores, 1)[:, 1].detach().cpu()
        # original implementation
        #logits = self.model(**kwargs.to('cuda')).logits[:, -1, :]
        # true_vector = logits[:, self.token_true_id]
        # false_vector = logits[:, self.token_false_id]
        # batch_scores = torch.stack([false_vector, true_vector], dim=1)
        # batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)        
        # scores = batch_scores[:, 1].exp().tolist()
        return {
                "score": scores
            }