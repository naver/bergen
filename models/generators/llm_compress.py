'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoTokenizer,  BitsAndBytesConfig
import torch
from models.generators.generator import Generator
import random
from autocompressor.auto_compressor import LlamaAutoCompressorModel

random.seed(42)
class LLM(Generator):
    def __init__(self, 
                model_name=None, 
                max_new_tokens=1, 
                max_doc_len=100,
                max_length=None,
                prompt=None,
                quantization=None,
                attn_implementation="flash_attention_2"
                 ):


        self.max_length = max_length
        self.model_name = model_name
        self.max_doc_len = max_doc_len
        self.quantization = quantization

        tokenizer_name = self.model_name
        model_class = LlamaAutoCompressorModel
        self.tokenizer = None
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tok.padding_side = "left"
        self.tok.pad_token = self.tok.bos_token

        if quantization == "int8":
            quant_config = BitsAndBytesConfig(
                llm_int8_enable_fp32_cpu_offload=True
            )
            self.model = model_class.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )


        elif quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )

            self.model = model_class.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
        else:
            self.model = model_class.from_pretrained(
                self.model_name,
                device_map='auto',
            )

        # self.model.merge_and_unload()
        #self.model.config.use_cache = False
        self.model = self.model.bfloat16()
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt

    def get_response(self):
        return '\nResponse:\n'

    def get_response_template_ids(self):
        response_template =  self.get_response()
        return self.tokenizer.encode(response_template, add_special_tokens=False)

    def prediction_step(self, model, model_input,label_ids=None):
        pass

    def generate(self, str_input):

        prompt, docs = str_input

        prompt_tokens = self.tok(prompt, add_special_tokens=False,padding=True, return_tensors="pt").input_ids.cuda()
        context_tokens = self.tok(docs, add_special_tokens=False, padding=True,return_tensors="pt").input_ids.cuda()

        summary_vectors = self.model(context_tokens, output_softprompt=True).softprompt

        generation_with_summary_vecs = self.model.generate(prompt_tokens, do_sample=False, softprompt=summary_vectors, max_new_tokens=self.max_new_tokens)
        prompt_len = prompt_tokens.size(1)
        generated_ids = generation_with_summary_vecs[:, prompt_len:]
        decoded = self.tok.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded



    def collate_fn(self, examples, eval=False, **kwargs):
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e)[1] for e in examples]
        question = [self.format_instruction(e)[0] for e in examples]

        label = [e['label'] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)

        data_dict = {}
        # for inference just format and tokenize instruction 
        model_input =  question
        
        data_dict.update({
            'model_input': (model_input,instr),
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
            compiled_prompt =  question , docs
        else:
            # without retrieval we don't put documents in the prompt
            #compiled_prompt = (question+ self.get_response(),None) #self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs, question)
            compiled_prompt = (question + self.get_response(), None) #self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs, question)
        return compiled_prompt 
    