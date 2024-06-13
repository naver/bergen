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


class LLM:
    def __init__(self, model_name, batch_size=1, custom_format_instruction=None, pos_word=None, neg_word=None):
        self.batch_size = batch_size
        self.custom_format_instruction = custom_format_instruction

        self.model_name = model_name

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
        if 'system' in self.tokenizer.chat_template:
            prefix =  [{'role': 'system',
                'content': "You are an evaluation tool. Just answer by {{Yes}} or {{No}}."}]
            prefix.extend([{'role': 'user',
                'content': f"Here is a question, a golden answer and an AI-generated answer. Can you judge whether the AI-generated answer is correct according to the question and golden answer, simply answer Yes or No.\n Question: {question}. \ngolden answer: {answer} \n Generated answer: {prediction}"}
                ]
                )            
        else:
            prefix = [{'role': 'user',
                'content': f"You are an evaluation tool. Just answer by {{Yes}} or {{No}}. Here is a question, a golden answer and an AI-generated answer. Judge whether the AI-generated answer is correct according to the question and golden answer, answer with {{Yes}} or {{No}}.\nQuestion: {question}.\nGolden answer: {answer}\nGenerated answer: {prediction}"}
                ]
        return self.tokenizer.apply_chat_template(prefix,  add_generation_prompt=True, tokenize=False) +'Response: {'

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
            scores.extend([ 1 if "yes" in rep.lower() else 0 for rep in decoded ])
            weird.extend([ 1 if ("no" not in rep.lower() and "yes" not in rep.lower()) else 0 for rep in decoded ])
            tq.set_description(f"score: {np.mean(scores)* 100:4.1f}%: weird {np.mean(weird)* 100:4.1f}%")
        return np.mean(scores), scores

