'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license

Tokenized_Sorted_Dataset class to format instruction, tokenize, sort data.
'''

from torch.utils.data import Dataset
from tqdm import tqdm
import random


class Tokenized_Sorted_Dataset(Dataset):
    """
    A dataset class to format instruction, tokenize, and sort data.
    """

    def __init__(self, data, model, training=False):

        self.model = model
        self.tokenizer = self.model.tokenizer
        self.training =  training

        # Preprocess, tokenize and store lengths
        processed_data = []
        if self.training:
            for item in tqdm(data):
                formatted_instr = self.model.format_instruction(item) + (item['label'] if isinstance(item['label'], str) else random.choice(item['label'])) + self.tokenizer.eos_token
                item['formatted_instruction'] = formatted_instr
                tokenized_input = self.tokenizer(formatted_instr, truncation=True, return_tensors="pt")
                length = tokenized_input['input_ids'].size(1)  # Length of tokenized input
                processed_data.append((length, item, tokenized_input))
        else:
            for item in tqdm(data):
                formatted_instr = self.model.format_instruction(item)
                item['formatted_instruction'] = formatted_instr
                tokenized_input = self.tokenizer(formatted_instr, truncation=True, return_tensors="pt")
                length = tokenized_input['input_ids'].size(1)
                processed_data.append((length, item, tokenized_input))

            # Sort by tokenized input length
        self.sorted_data = sorted(processed_data, key=lambda x: x[0])


    def __len__(self):
        return len(self.sorted_data)

    def __getitem__(self, idx):
        _, item, tokenized_input = self.sorted_data[idx]
        # Update the item with tokenized input for consistency
        item['tokenized_input'] = tokenized_input
        return item

    def select(self, indices):
        # Create a new dataset based on selected indices
        selected_data = []
        for i in indices:
            _, item, tokenized_input = self.sorted_data[i]
            item['tokenized_input'] = tokenized_input  # Ensure tokenized_input is included
            selected_data.append(item)
        return selected_data