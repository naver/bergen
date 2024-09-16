'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

import torch
from models.generators.generator import Generator
import random
random.seed(42)

class APIBenchReturnRetrievalApi(Generator):
    # this class returns the retrieval (specific to APIBench dataset but this class can be generalized)
    def __init__(self, 
                 model_name=None, 
                 batch_size=1,
                 **kwargs
                 ):
        Generator.__init__(self, model_name=model_name, batch_size=batch_size)

    def tokenizer(self, instr, **kwargs):
        return {'instruction': instr, 'input_ids': torch.Tensor([[]])}
    
    def get_response(self):
        return '\nResponse:\n'
    
    def format_instruction(self, sample):
        # in case we have previously retrieved documents
        if 'doc' in sample:
            docs = ''
            for i, doc in enumerate(sample['doc']):
                doc = ' '.join(doc.split())
                docs += f"Document {i+1}: {doc}\n"
            compiled_prompt = docs
        else:
            compiled_prompt = ""
        return compiled_prompt + self.get_response()
    
    def prediction_step(self, model, model_input, label_ids=None):
        """Just to re-define the abstract class"""
        return None

    def generate(self, inp):
        """
        Example substring of the retrieval: 'api_call: T5ForConditionalGeneration.from_pretrained('pszemraj/long-t5-tglobal-base-16384-book-summary');'
        """
        if isinstance(inp, list): # batched input even for batch size == 1
            try:
                return [inpp.split('api_call: ')[1].split(';')[0] for inpp in inp]
            except:
                print(inp)
                raise RuntimeError("error getting retrieved api call... make sure to activate retrieval and the format is 'api_call: [...];'   ")

    def collate_fn(self, examples, eval=False):
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e) for e in examples]
        label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
        return {
            'model_input': instr,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        }