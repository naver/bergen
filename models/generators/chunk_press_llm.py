'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import torch
from models.generators.generator import Generator
import warnings
from utils import prepare_labels
from peft import AutoPeftModelForCausalLM, PeftConfig
import random
import os
import json
import gc
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from kvpress import ExpectedAttentionPress, ChunkPress
from kvpress.pipeline import KVPressTextGenerationPipeline
from jinja2.exceptions import TemplateError


random.seed(42)


class  BergenKVPressTextGenerationPipeline(KVPressTextGenerationPipeline):
    def preprocess(
            self,
            context: str,
            questions: list[str],
            answer_prefix: str,
            max_context_length: int): # just to avoid the chat template: we control that
        # Tokenize the context and questions
        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) for question in questions]
        
        print('context ids', context_ids.size())
        print('questions ids', [question.size() for question in question_ids])

        return {"context_ids": context_ids, "questions_ids": question_ids}
    
    
PIPELINE_REGISTRY.register_pipeline(
    "bergen-kv-press-text-generation",
    pipeline_class=BergenKVPressTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)


class ChunkPressLLM(Generator):
    def __init__(self, 
                model_name=None,
                batch_size=1, 
                max_new_tokens=1, 
                max_doc_len=100,
                max_length=None,
                prompt=None,
                attn_implementation="flash_attention_2",
                device_map='auto',
                compression_ratio=0.5
                ):
        Generator.__init__(self, model_name=model_name, batch_size=batch_size)
        if not "A100" in torch.cuda.get_device_name(torch.cuda.current_device):
            attn_implementation="sdpa"
        self.max_length = max_length
        self.max_doc_len = max_doc_len
         # get tokenizer of lora adapter if exists else use models' tokenizer

        tokenizer_name = self.model_name
        print(tokenizer_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            config_dict = os.path.join(tokenizer_name, 'config.json')
            with open(config_dict, 'r') as f:
                config = json.load(f)
            tokenizer_name = config['_name_or_path']
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = (
            self.tokenizer.bos_token
            or self.tokenizer.pad_token
            or self.tokenizer.eos_token
        )
        
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        
        model_kwargs = {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}
        self.pipe = pipeline("bergen-kv-press-text-generation", model=self.model_name, device="cuda", model_kwargs=model_kwargs)

        # context = "A very long text you want to compress once and for all"
        # question = "\nA question about the compressed context"  # optional
        # answer = pipe(context, question=question, press=press)["answer"]

        self.press = ChunkPress(press=ExpectedAttentionPress(compression_ratio=compression_ratio), chunk_length=256)
        
    def get_response(self):
        return '\nResponse:\n'

    def get_response_template_ids(self):
        response_template =  self.get_response()
        return self.tokenizer.encode(response_template, add_special_tokens=False)

    def prediction_step(self, model, model_input, label_ids=None):
        output = model(**model_input, labels=label_ids)
        return output.logits, output.loss
       
    def generate(self, model_input):
        # print('######################@')
        contexts = model_input['context']
        questions = model_input['question']
        answers = []
        for context, question in zip(contexts, questions):
            # print('###context',context)
            # print('###question', question)
            print('YO', context, question)
            answer = self.pipe(context, question=[question], press=self.press)['answer']
            
            # print('###answer',answer)
            answers.append(answer)
            
        return answers

    def __del__(self):
    #del self.model.llm_engine.model_executor
    #    del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def collate_fn(self, examples, eval=False, **kwargs):
        q_ids = [e['q_id'] for e in examples]

        label = [e['label'] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
        instr = [e["formatted_instruction"] for e in examples]

        # Let's form the inputs to the pipeline: context and question. 
        # The context is all our prompt until the end of the documents:
        instructions = [e['formatted_instruction'] for e in examples]
        split_indices = [elt.find('\n\n\nQuestion:') for elt in instructions]
        for instr, index in zip(instructions, split_indices):
            assert index > 0, f"{index} {instr}"
        contexts = [instr[:split_index] for instr, split_index in zip(instructions, split_indices)]
        questions = [instr[split_index:] for instr, split_index in zip(instructions, split_indices)]

        model_input = {
            'context': contexts,
            'question': questions
        }

        assert eval

        data_dict = {
            'model_input': model_input,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        }

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
        return compiled_prompt + self.get_response()

    def compile_prompt(self, system_prompt, user_prompt, question, docs=None):
        # check if chat template allows for system prompts

        # if has chat_template e.g. gamma does not use it
        if self.tokenizer.chat_template is None:
            user_prompt_with_values = eval(user_prompt).replace(':\ ', ': ')
            return f"{system_prompt}\n{user_prompt_with_values}"
        else:
            # We try using the chat template with a system
            # Sometimes system not supported: we catch it.
            try:
                instr_prompt = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": eval(user_prompt).replace(':\ ', ': ')}
                ]
                return self.tokenizer.apply_chat_template(instr_prompt,  add_generation_prompt=True, tokenize=False)
            
            except TemplateError as e:
                if "System role not supported" in str(e):
                    user_prompt_with_values = eval(user_prompt).replace(':\ ', ': ')
                    instr_prompt = [
                        {"role": "user", "content": f"{system_prompt}\n{user_prompt_with_values}"}
                    ]    
                    return self.tokenizer.apply_chat_template(instr_prompt,  add_generation_prompt=True, tokenize=False)
                else:
                    raise e