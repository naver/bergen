'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

import openai
import os
from models.generators.generator import Generator


class OpenAI(Generator):
    def __init__(self, 
                model_name="gpt-3.5-turbo", 
                max_new_tokens=1, 
                max_doc_len=100,
                max_length=None,
                prompt=None
                 ):
        self.client = openai.OpenAI(api_key = os.environ.get("OPENAI_API_KEY"),)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.chat_template = True
        self.total_cost = 0
        self.prompt_cost = 0
        self.completion_cost = 0
        self.max_doc_len = max_doc_len
        self.max_length = max_length
        self.tokenizer=None

    def generate(self, messages):
        responses=[]
        for msg in messages:
            response = self.client.chat.completions.create( messages=msg, model=self.model_name)
            responses.append(response.choices[0].message.content)
            t,p,c = self.openai_api_calculate_cost(response.usage)
            self.total_cost += t
            self.prompt_cost += p 
            self.completion_cost += c 

        return responses


    def openai_api_calculate_cost(self,usage):
        pricing = {
            'gpt-3.5-turbo': {
                'prompt': 0.0015 ,
                'completion': 0.0020,
            },
            'gpt-4-1106-preview': {
                'prompt': 0.01,
                'completion': 0.03,
            },
            'gpt-4': {
                'prompt': 0.03,
                'completion': 0.06,
            },
            'gpt-4-0125-preview':{
                'prompt': 0.01,
                'completion': 0.03,                
            },
            'gpt-4o': {
            'prompt': 0.005,  #US$5.00 / 1M tokens
            'completion': 0.015,  #US$15.00 / 1M tokens
            }                 
        }

        try:
            model_pricing = pricing[self.model_name]
        except KeyError:
            raise ValueError("Invalid model specified")

        prompt_cost = usage.prompt_tokens * model_pricing['prompt'] / 1000
        completion_cost = usage.completion_tokens * model_pricing['completion'] / 1000

        total_cost = prompt_cost + completion_cost
        # round to 6 decimals
        total_cost = round(total_cost, 6)

        #print(f"\nTokens used:  {usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} completion = {usage.total_tokens:,} tokens")
        #print(f"Total cost for {model}: ${total_cost:.4f}\n")

        return (total_cost,prompt_cost,completion_cost)

    # only required for training
    def prediction_step(self, model, model_input, label_ids=None):
        # e.g.       
        # output = model(**model_input, labels=label_ids)
        # return output.logits, output.loss
        pass
    
    def collate_fn(self, examples, eval=False, **kwargs):
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e) for e in examples]

        label = [e['label'] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)

        data_dict = {}
        # for inference just format and tokenize instruction 
        instr = [self.format_instruction(e) for e in examples]
        model_input =  instr
        
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
            compiled_prompt = self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs,question)
        return compiled_prompt        



    def compile_prompt(self, system_prompt, user_prompt, question, docs=None):
        # check if chat template allows for system prompts

        # if has chat_template e.g. gamma does not use it
        if self.chat_template == None:
            user_prompt_with_values = eval(user_prompt).replace(':\ ', ': ')
            return f"{system_prompt}\n{user_prompt_with_values}"
        else:
            if True: #'system' in self.tokenizer.chat_template:
                instr_prompt = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": eval(user_prompt).replace(':\ ', ': ')}
                ]
            # if no system prompts are allowed just append system prompt to user prompt
            else:
                user_prompt_with_values = eval(user_prompt).replace(':\ ', ': ')
                instr_prompt = [
                    {"role": "user", "content": f"{system_prompt}\n{user_prompt_with_values}"}
                ]    
            return instr_prompt #self.tokenizer.apply_chat_template(instr_prompt, tokenize=False)