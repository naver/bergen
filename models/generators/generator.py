'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from abc import ABC, abstractmethod

class Generator(ABC):
    def __init__(self, model_name=None):
        self.model_name = model_name

    @abstractmethod
    def generate(self, inp):
        pass
    
    @abstractmethod
    def collate_fn(self, inp):
        pass

    # only required for training
    @abstractmethod
    def prediction_step(self, model, model_input, label_ids=None):
        # e.g.       
        # output = model(**model_input, labels=label_ids)
        # return output.logits, output.loss
        pass

    def compile_prompt(self, system_prompt, user_prompt, question, docs=None):
        # check if chat template allows for system prompts

        # if has chat_template e.g. gamma does not use it
        if self.tokenizer.chat_template == None:
            user_prompt_with_values = eval(user_prompt).replace(':\ ', ': ')
            return f"{system_prompt}\n{user_prompt_with_values}"
        else:
            if 'system' in self.tokenizer.chat_template:
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
            return self.tokenizer.apply_chat_template(instr_prompt,  add_generation_prompt=True, tokenize=False)


        

