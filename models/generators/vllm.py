'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoTokenizer
import torch
from models.generators.generator import Generator
import random

from vllm import LLM as vllm
from vllm import  SamplingParams
import gc

random.seed(42)


class LLM(Generator):
    def __init__(self,
                model_name=None, 
                batch_size=1,
                max_new_tokens=1, 
                max_doc_len=100,
                max_length=None,
                prompt=None,
                quantization=None,
                gpu_memory_utilization=0.9
                ):
        Generator.__init__(self, model_name=model_name, batch_size=batch_size)

        self.quantization = quantization
        self.max_length = max_length
        self.max_doc_len = max_doc_len
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token

        if self.quantization is None:
            self.model = vllm(model=self.model_name,tensor_parallel_size=torch.cuda.device_count(),dtype=torch.float16,gpu_memory_utilization=gpu_memory_utilization,max_model_len=self.max_length,enforce_eager=True,kv_cache_dtype="fp8")        
        else:
            self.model = vllm(model=self.model_name,tensor_parallel_size=torch.cuda.device_count(),gpu_memory_utilization=gpu_memory_utilization,quantization=self.quantization)
        self.sampling_params =  SamplingParams(temperature=1,max_tokens=max_new_tokens,best_of=1, top_p=1, top_k=-1)




    def prediction_step(self, model, model_input, label_ids=None):
        output = model(**model_input, labels=label_ids)
        return output.logits, output.loss

    def generate(self, instr_tokenized):
        outputs = self.model.generate(instr_tokenized, self.sampling_params)
        decoded = [output.outputs[0].text for output in outputs]
        return decoded

    def collate_fn(self, examples, eval=False, **kwargs):
        ignore_index = -100
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e) for e in examples]

        label = [e['label'] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)

        data_dict = {}

        # for inference just format and tokenize instruction 
        model_input = [self.format_instruction(e) for e in examples]
        
        data_dict.update({
            'model_input': model_input,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        })

        return data_dict

    def format_instruction(self, sample):
        # will be injected into formatted prompt string
        question = sample['query']
        # in case we have previously retrieved documents
        if 'doc' in sample:
            docs = ''
            for i, doc in enumerate(sample['doc']):
                doc = ' '.join(doc.split()[:self.max_doc_len])
                docs += f"Document {i+1}: {doc}\n"
            compiled_prompt = self.compile_prompt(self.prompt.system, self.prompt.user, question, docs)
        else:
            # without retrieval we don't put documents in the prompt
            compiled_prompt = self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs, question)

        return compiled_prompt
