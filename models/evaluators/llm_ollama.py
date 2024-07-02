'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import torch
import re
import numpy as np
from langchain_community.llms import Ollama
import ollama
from langchain_core.prompts import ChatPromptTemplate
import omegaconf


class LLM:
    def __init__(self, model_name, batch_size=1, custom_format_instruction=None, pos_word="Yes", neg_word="No", prompt="default_prompt", basic_url="http://localhost:11434"):
        self.batch_size = batch_size
        self.custom_format_instruction = custom_format_instruction
        self.pos_word = pos_word 
        self.neg_word = neg_word 
        self.prompt = omegaconf.OmegaConf.load(f"config/evaluator/{prompt}.yaml")['prompt']
        self.model_name = model_name
        self.model = Ollama(model=model_name, base_url=f"{basic_url}", system=eval(self.prompt.system))
       



    def create_instruction(self,sample):
        answer = sample['reference']
        question=sample['question']
        prediction=sample['candidate']
        pos_word = self.pos_word
        neg_word = self.neg_word
        template = ChatPromptTemplate.from_messages([
            ("user", eval(self.prompt.user)),
            ("ai", ' Response: {{'),
        ])
       
        return template.invoke({'answer':answer, 'question':question, 'prediction':prediction, 'pos_word': self.pos_word, 'neg_word': self.neg_word})

   

    @torch.no_grad()
    def __call__(self, predictions, references, questions):
        # Loading the TensorFlow Hub model
        assert len(predictions) == len(references) == len(questions)
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        instrs = [self.create_instruction(sample) if self.custom_format_instruction == None else self.custom_format_instruction(sample) for sample in examples]
        scores = list()
       
        for i in (tq:=tqdm(range(0, len(instrs), self.batch_size), desc=f'LLM evaluation with {self.model_name}...')):
            outputs = self.model.batch(instrs[i:i+self.batch_size])
            scores.extend([ 1 if self.pos_word.lower() in rep.lower() else 0 for rep in outputs ])
            tq.set_description(f" score: {np.mean(scores)* 100:4.1f}%")
        return np.mean(scores), scores
        
