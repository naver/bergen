'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import numpy as np
from vllm import LLM as vllm
from vllm import  SamplingParams
import omegaconf



class LLM:
    """
    - relies on vllm for inference, directly loads the model and runs inference (no need to initiate vllm server in advance) 
    - output score for each sample is 1 (when positive word is present in llm output) or 0  (otherwise) 
    """
    def __init__(self, model_name, batch_size=1, custom_format_instruction=None, pos_word='Yes', neg_word='No', prompt="default_prompt"):
        self.batch_size = batch_size
        self.custom_format_instruction = custom_format_instruction
        self.pos_word = pos_word  
        self.neg_word = neg_word 
        self.model_name = model_name
        self.prompt = omegaconf.OmegaConf.load(f"config/evaluator/{prompt}.yaml")['prompt']
        self.system_prompt = eval(self.prompt.system)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.quantization = None
        if self.quantization is None:
            self.model = vllm(model=self.model_name,tensor_parallel_size=torch.cuda.device_count(),gpu_memory_utilization=0.9,max_model_len=4096,enforce_eager=False,kv_cache_dtype="fp8")        
        else:
            self.model = vllm(model=self.model_name,tensor_parallel_size=torch.cuda.device_count(),quantization=self.quantization)
        self.sampling_params =  SamplingParams(temperature=0,max_tokens=10,best_of=1, top_p=1, top_k=-1)


    def create_instruction(self,sample):
        answer = sample['reference']
        question=sample['question']
        prediction=sample['candidate']
    
        prefix=[]
        if 'system' in self.tokenizer.chat_template:
            prefix =  [{'role': 'system',
                'content': self.system_prompt}]
        prefix.extend([{'role': 'user',
            'content': eval(self.prompt.user).replace(":\ ", ": ")}]
            )
        prefix.extend([{'role': 'assistant',
            'content': eval(self.prompt.assistant).replace(":\ ", ": ")}]
            )

        return self.tokenizer.apply_chat_template(prefix,  add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def __call__(self, predictions, references, questions):
        # Loading the TensorFlow Hub model
        assert len(predictions) == len(references) == len(questions)
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        instrs = [self.create_instruction(sample) if self.custom_format_instruction == None else self.custom_format_instruction(sample) for sample in examples]
        scores = list()
        weird = list() 
        # Perform batch inference
        for i in (tq:=tqdm(range(0, len(instrs), self.batch_size), desc=f'LLM evaluation with {self.model_name}...')):
            outputs = self.model.generate(instrs[i:i+self.batch_size], self.sampling_params)
            decoded = [output.outputs[0].text for output in outputs]
            scores.extend([ 1 if self.pos_word.lower() in rep.lower() else 0 for rep in decoded ])
            weird.extend([ 1 if (self.neg_word.lower() not in rep.lower() and self.pos_word not in rep.lower()) else 0 for rep in decoded ])
            tq.set_description(f"score: {np.mean(scores)* 100:4.1f}%: weird {np.mean(weird)* 100:4.1f}%")
        return np.mean(scores), scores

