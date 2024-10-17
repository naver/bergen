from transformers import GenerationConfig
from tqdm import tqdm
import torch
import gc
import re
import numpy as np
from hydra.utils import instantiate
from torch.distributions import Categorical
from difflib import SequenceMatcher
from collections import defaultdict

class LLM_LL():
    def __init__(self, generator_config, prompt):
        #generator_config['init_args']['attn_implementation'] = 'sdpa'
        self.llm = instantiate(generator_config['init_args'], prompt=prompt)
        self.generation_config = GenerationConfig.from_model_config(self.llm.model.config) 
        self.generation_config.do_sample=False,
        self.generation_config.output_logits=True 
        self.generation_config.return_dict_in_generate=True  
        self.generation_config.max_new_tokens=5
        
        
    def collate_fn(self, samples):
        #decompotes prompt into different subparts, and keep trace of subparts position (to quantify attention at these subparts)
        #prompt_len = [len(self.llm.tokenizer(sample['instruction'], add_special_tokens=False)['input_ids']) for sample in samples ]

        tokenized = self.llm.tokenizer([sample['instruction'] + sample['candidate'] for sample in samples], padding=True, truncation=True,  add_special_tokens=False, return_tensors="pt")
        tokenized_label = self.llm.tokenizer([sample['instruction'] + sample['label'][0] for sample in samples], padding=True, truncation=True,  add_special_tokens=False, return_tensors="pt")
        
        return tokenized, tokenized_label
    
    @torch.no_grad()
    def __call__(self, predictions, references, questions, instructions):
        assert len(predictions) == len(references) == len(questions) == len(instructions)
        examples = [{'question': questions[i], 'candidate': predictions[i], 'label': references[i], 'instruction': instructions[i]}  for i in range(len(predictions))]
        # The outputs are raw logits.
        scores = list()
        # Perform batch inference
        for i in (tq:=tqdm(range(0, len(examples), self.llm.batch_size), desc=f'LL estimate with {self.llm.model_name}...')):
            # Extract batch
            batch_examples = examples[i:i+self.llm.batch_size]
            inputs, gold_inputs = self.collate_fn(batch_examples)
            inputs = inputs.to(self.llm.model.device)
            pred_output = self.llm.model.forward(**inputs).logits
            breakpoint()
            gen_ids = inputs['input_ids'][:,-1].to(self.llm.model.device)
            pred_prob = torch.take(torch.softmax(pred_output[:,-1], 1), gen_ids)
            breakpoint()
            gold_inputs = gold_inputs.to(self.llm.model.device)
            gold_output = self.llm.model.forward(**gold_inputs).logits
            gold_ids = gold_inputs['input_ids'][:,-1].to(self.llm.model.device)
            gold_prob = torch.take(torch.softmax(gold_output[0,-1], 0), gold_ids)
            
            if pred_prob>=gold_prob:
                breakpoint()
            scores.append(float(pred_prob>=gold_prob))    
            
        return np.mean(scores), scores

