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
    def __init__(self, model_config: dict, batch_size: int = None, config: str = "default_qa" ):
        """
            model_config: generator config specified as yaml file in cofig/generator directory
            batch_size: if none, it keeps default llm batch size from config 
            confg: name of evaluator config specified as yaml file at config/evaluators
        """
        eval_config = omegaconf.OmegaConf.load(f"config/evaluator/{config}.yaml")
        model_config['init_args']['max_new_tokens']= eval_config['max_new_tokens']
        self.llm = instantiate(model_config['init_args'], prompt=eval_config['prompt'])
        self.options = eval_config.output_options
        self.rubrik_section = ", ".join(["{"+opt+"}" for opt in self.options])
        self.prompt = eval_config['prompt']
        self.llm.sampling_params.max_new_token = eval_config['max_new_tokens']
        if not batch_size == None:
            self.llm.batch_size = batch_size
        self.llm.max_new_tokens = eval_config['max_new_tokens']
        self.system_prompt = self.prompt.system.format(rubrik_section=self.rubrik_section)
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
                'content': self.prompt.user.format(rubrik_section=self.rubrik_section,
                                                    question=question,
                                                    answer=answer,
                                                    prediction=prediction)}]
            )
        
        else:
            prefix = ([{'role': 'user',
                'content': self.prompt.user_without_system.format(rubrik_section=self.rubrik_section,
                                                    question=question,
                                                    answer=answer,
                                                    prediction=prediction)}]
            )
        if 'assistant' in self.prompt:
            prefix.extend([{'role': 'assistant',
                'content': self.prompt.assistant}]
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
            tq.set_description(f" score: {get_mean_without_unknown(scores)* 100:4.1f}%, weird :{float(len(weird))/len(scores)*100:4.1f}%")
        logger.info(weird)
        print("Weird", len(weird))
    
        return get_mean_without_unknown(scores), scores

