'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from abc import ABC, abstractmethod
from modules.dataset import Tokenized_Sorted_Dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Generator(ABC):
    def __init__(self, model_name=None, batch_size=1):
        self.model_name = model_name
        self.batch_size = batch_size

    @abstractmethod
    def generate(self, inp):
        raise NotImplementedError
    
    @abstractmethod
    def collate_fn(self, inp):
        raise NotImplementedError

    def eval(self, dataset):
        if self.tokenizer:
            tokenized_and_sorted_dataset = Tokenized_Sorted_Dataset(dataset, self, training=False)
            dataloader = DataLoader(tokenized_and_sorted_dataset, batch_size=self.batch_size, collate_fn=lambda l: self.collate_fn(l, eval=True), num_workers=4)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda l: self.collate_fn(l, eval=True), num_workers=4)
        
        responses, instructions, query_ids, queries, labels, ranking_labels = list(), list(), list(), list(), list(), list()
        for data_dict in tqdm(dataloader, desc='Generating'):
            id_ = data_dict['q_id']
            instruction = data_dict['instruction']
            query_ids += id_
            label = data_dict['label']
            labels += label
            queries += data_dict['query']
            ranking_labels += data_dict['ranking_label']
            instructions += instruction
            generated_response = self.generate(data_dict['model_input'])
            responses += generated_response
        return query_ids, queries, instructions, responses, labels, ranking_labels


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
