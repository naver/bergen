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
from models.evaluators.utils import *


class OllamaEval:
    """
    - uses existing ollama server for inference, need to specify the url of the server via basic_url parameter 
    - output score for each sample is 1 (when positive word is present in llm output) or 0  (otherwise) 
    """
    def __init__(self, model_name, batch_size=1, config="default_qa", basic_url="http://localhost:11434"):
        eval_config = omegaconf.OmegaConf.load(f"config/evaluator/{config}.yaml")
        self.batch_size = batch_size
        self.options = eval_config.output_options
        self.prompt = eval_config['prompt']
        self.max_new_tokens = eval_config['max_new_tokens']
        self.batch_size = batch_size
        self.rubrik_section = "\n - ".join(sorted([f"{opt} answer" for opt in self.options]))
        self.system_prompt = eval(self.prompt.system)
        self.output_values = torch.tensor([self.options[opt] for opt in sorted(self.options)]).float()
        self.model_name = model_name
        self.model = Ollama(model=model_name, base_url=f"{basic_url}", system=eval(self.prompt.system))
       

    def create_instruction(self,sample):
        answer = ", ".join(sample['reference'])
        question=sample['question']
        #need to remove "{}"
        prediction=sample['candidate'].replace("{", "").replace("}","")
        template = ChatPromptTemplate.from_messages([
            ("user", eval(self.prompt.user)),
            ("ai", ' Response: '),
        ])
        return template.invoke({'answer':answer, 'question':question, 'prediction':prediction, 'self.rubrik_section':self.rubrik_section})

   

    @torch.no_grad()
    def __call__(self, predictions, references, questions):
        # Loading the TensorFlow Hub model
        assert len(predictions) == len(references) == len(questions)
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        instrs = [self.create_instruction(sample) for sample in examples]
        scores = list()
        weird = list()
       
        for i in (tq:=tqdm(range(0, len(instrs), self.batch_size), desc=f'LLM evaluation with {self.model_name}...')):
            outputs = self.model.batch(instrs[i:i+self.batch_size])
            batch_scores, batch_weird  = process_llm_outputs_assess_scores(outputs, self.options)
            scores.extend(batch_scores)
            weird.extend(batch_weird)
            tq.set_description(f" score: {get_mean_without_unknown(scores)* 100:4.1f}%, weird :{float(len(weird))/len(scores)*100:4.1f}%")
        return get_mean_without_unknown(scores), scores
        
