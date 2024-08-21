'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from tqdm import tqdm
import torch
import numpy as np
from vllm import LLM as vllm
from vllm import  SamplingParams
import omegaconf
from hydra.utils import instantiate
import random
from models.evaluators.utils import *
import logging
logger = logging.getLogger(__name__)
import gc


class VLLMeval:
    """
    - relies on vllm for inference, directly loads the model and runs inference (no need to initiate vllm server in advance) 
    - output score for each sample is 1 (when positive word is present in llm output) or 0  (otherwise) 
    """
    def __init__(self, model_config, batch_size=1, pos_word='Yes', neg_word='No', config="default_qa"):
        eval_config = omegaconf.OmegaConf.load(f"config/evaluator/{config}.yaml")
        model_config = omegaconf.OmegaConf.load(f"config/generator/vllm_{model_config}.yaml")
        self.llm = instantiate(model_config['init_args'], prompt=eval_config['prompt'])
        self.options = eval_config.output_options
        self.rubrik_section = "\n - ".join([f"{self.options[opt]} for {opt} answer" for opt in self.options])

        self.prompt = eval_config['prompt']
        self.llm.sampling_params.max_new_token = eval_config['max_new_tokens']
        self.llm.batch_size = batch_size
        self.llm.max_new_tokens = eval_config['max_new_tokens']
        self.system_prompt = eval(self.prompt.system)
        self.output_ids = [self.llm.tokenizer.encode(opt, add_special_tokens=False)[-1] for opt in sorted(self.options)]
        self.output_values = torch.tensor([self.options[opt] for opt in sorted(self.options)]).float()
        
     
    def create_instruction(self,sample):
        answer = sample['reference']
        question=sample['question']
        prediction=sample['candidate']
        if 'response' in sample:
            response = sample['response']
        else:
            response = None
        prefix = []
        if 'system' in self.llm.tokenizer.chat_template:
            prefix =  [{'role': 'system',
                'content': self.system_prompt}]
            prefix.extend([{'role': 'user',
                'content': eval(self.prompt.user)}]
            )
        
        else:
            prefix = ([{'role': 'user_without_system',
                'content': eval(self.prompt.user)}]
            )
        if 'assistant' in self.prompt:
            prefix.extend([{'role': 'assistant',
                'content': eval(self.prompt.assistant)}]
                )
        if not response is None:
            prefix.extend([{'role': 'assistant',
                'content': response}]
            )
        return self.llm.tokenizer.apply_chat_template(prefix,  add_generation_prompt=True, tokenize=False) 

    def __del__(self):
    #    logger.info("Deleting object")        
        torch.cuda.empty_cache()
        gc.collect()        
    
    @torch.no_grad()
    def __call__(self, predictions, references, questions):
        # Loading the TensorFlow Hub model
        assert len(predictions) == len(references) == len(questions)
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        instrs = [self.create_instruction(sample) for sample in examples]
        scores = list()
        weird = list() 
        # Perform batch inference
        for i in (tq:=tqdm(range(0, len(instrs), self.llm.batch_size), desc=f'LLM evaluation with {self.llm.model_name}...')):
            decoded = self.llm.generate(instrs[i:i+self.llm.batch_size])
            batch_scores, batch_weird  = process_llm_outputs_assess_scores(decoded, self.options)
            scores.extend(batch_scores)
            weird.extend(batch_weird)
            #scores.extend([ 1 if self.pos_word.lower() in rep.lower() else 0 for rep in decoded ])
            #weird.extend([ 1 if (self.neg_word.lower() not in rep.lower() and self.pos_word not in rep.lower()) else 0 for rep in decoded ])
            tq.set_description(f" score: {get_mean_without_unknown(scores)* 100:4.1f}%, weird :{float(len(weird))/len(scores)*100:4.1f}%")
        logger.info(weird)
    
        return get_mean_without_unknown(scores), scores

