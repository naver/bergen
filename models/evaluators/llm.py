'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import omegaconf
from tqdm import tqdm
import torch
import re
import numpy as np
from hydra.utils import instantiate
from models.evaluators.utils import *

class LLMeval():
    """
    - relies on default HF inference 
    - output score is a floating number corresponding to the logit score output by model.generate for pos_word
    """
    #FIXME: we could reuse init from llm generator, but need to update max_length, max_new_token
    def __init__(self, model_config, batch_size=1, config="default_qa"):
        #model_config['init_args']['_target_'] = 'models.evaluators.llm.LLMeval'
        model_config = omegaconf.OmegaConf.load(f"config/generator/{model_config}.yaml")            
        eval_config = omegaconf.OmegaConf.load(f"config/evaluator/{config}.yaml")
        self.use_logits = eval_config.use_logits
        self.llm = instantiate(model_config['init_args'], prompt=eval_config['prompt'])
        self.options = eval_config.output_options
        self.prompt = eval_config['prompt']
        self.llm.max_new_token = eval_config['max_new_tokens']
        self.llm.batch_size = batch_size
        self.system_prompt = eval(self.prompt.system)
        self.output_ids = [self.llm.tokenizer.encode(opt, add_special_tokens=False)[-1] for opt in sorted(self.options)]
        self.output_values = torch.tensor([self.options[opt] for opt in sorted(self.options)]).float()
        
    def create_instruction(self,sample):
        answer = sample['reference']
        question=sample['question']
        prediction=sample['candidate']
        prefix = []
        if 'system' in self.llm.tokenizer.chat_template:
            prefix =  [{'role': 'system',
                'content': self.system_prompt}]
        prefix.extend([{'role': 'user',
            'content': eval(self.prompt.user)}]
            )
        prefix.extend([{'role': 'assistant',
            'content': eval(self.prompt.assistant)}]
            )
        return self.llm.tokenizer.apply_chat_template(prefix,  add_generation_prompt=True, tokenize=False) 

   
   
    def collate_fn(self, examples, max_length=512):
        instr = [self.create_instruction(sample) for sample in examples]  # Add prompt to each text
        self.llm.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        instr_tokenized = self.llm.tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
        return instr_tokenized, instr

    @torch.no_grad()
    def __call__(self, predictions, references, questions):
        # Loading the TensorFlow Hub model
        assert len(predictions) == len(references) == len(questions)
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        # The outputs are raw logits.
        scores = list()
        weird = list()
        # Perform batch inference
        for i in tqdm(range(0, len(examples), self.llm.batch_size), desc=f'LLM evaluation with {self.llm.model_name}...'):
            # Extract batch
            batch_examples = examples[i:i+self.llm.batch_size]
            inputs, instrs = self.collate_fn(batch_examples)
           
            if self.use_logits:
                # continuous model output:
                model_scores = self.llm.generate_logits(inputs)
                #get processed logits from model outputs: expected shape (n_tokens, 1, vocab_size)
                model_scores = torch.stack(model_scores)
                #get scores corresponding to self.output_ids from the first generated tokens ()
                model_scores = model_scores[0, :, self.output_ids].float()
                #normalizing scores - getting probablity of each of predefined labesl
                pos_prob = torch.softmax(model_scores, 1).detach().cpu()
                #final score is computed as interpolation between prob of label and it's associated value (defined by options map in config): eg. p(x=yes)*1 + p(x=no)*0 
                for i, score in enumerate(pos_prob):
                    scores.append(torch.dot(score,self.output_values))
            else:
                # discrete model output            
                # get real answer generation
                model_generations = self.llm.generate(inputs)
                batch_scores, batch_weird  = process_llm_outputs_assess_scores(model_generations, self.options)
                scores.extend(batch_scores)
                weird.extend(batch_weird)
                # if string value specified in options is present in the generated output: assign corresponding score,
                # if multiple values are present: take maximum value
                scores.extend(scores)
        
        torch.cuda.empty_cache()
        return get_mean_without_unknown(scores), scores

