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

random.seed(42)


class VLLM(Generator):
    def __init__(self,
                model_name: str = None, 
                batch_size: int = 1,
                max_new_tokens: int = 1, 
                max_doc_len: int = 100,
                max_length: int = None,
                prompt: str = None,
                quantization: str = None,
                gpu_memory_utilization: float = 0.9,
                temperature: float = 1.,
                use_beam_search: bool = False,
                best_of: int = 1,
                sampling: bool = False
                ):
        Generator.__init__(self,
                           model_name=model_name,
                           batch_size=batch_size,
                           max_new_tokens=max_new_tokens,
                           max_doc_len=max_doc_len,
                           max_length=max_length)
        
        self.quantization = quantization
        self.prompt = prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token

        if self.quantization is None:
            self.model = vllm(model=self.model_name,
                              tensor_parallel_size=torch.cuda.device_count(),
                              dtype=torch.float16,
                              gpu_memory_utilization=gpu_memory_utilization,
                              max_model_len=self.max_length,
                              enforce_eager=True,
                              kv_cache_dtype="auto")        
        else:
            self.model = vllm(model=self.model_name,
                              tensor_parallel_size=torch.cuda.device_count(),
                              gpu_memory_utilization=gpu_memory_utilization,
                              quantization=self.quantization)
            
        if use_beam_search:
            assert temperature == 0, f'beam search requires temperature = 0, not {temperature}'
            if best_of == 1:
                Warning('You are doing beam search with best_of=1: it is greedy decoding. Consider increasing best_of.')
            self.sampling_params =  SamplingParams(temperature=temperature,
                                                   max_tokens=max_new_tokens,
                                                   best_of=best_of,
                                                   top_p=1,
                                                   top_k=-1)
        else:
            if best_of > 1:
                Warning('You set best_of > 1 without beam_search: vllm will do best of n sampling.')
                assert temperature > 0, 'To do best of n sampling, you need temperature > 0'
            self.sampling_params =  SamplingParams(temperature=0,
                                                   max_tokens=max_new_tokens,
                                                   best_of=best_of,
                                                   top_p=1,
                                                   top_k=-1)
            
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
        model_input = [self.format_instruction(e, eval=True)[0] for e in examples]
        
        data_dict.update({
            'model_input': model_input,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        })

        return data_dict
