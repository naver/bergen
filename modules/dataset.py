'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from torch.utils.data import Dataset
from tqdm import tqdm
import random


class Tokenized_Sorted_Dataset(Dataset):
    """
    Custom PyTorch Dataset that tokenizes and sorts data based on token length.

    Args:
        data (list): The input data to be processed.
        model: The model object that provides the tokenizer and formatting.
        training (bool): Whether to include labels in tokenization (for training).
    """
    def __init__(self, data, model, training: bool = False):

        self.model = model
        self.tokenizer = self.model.tokenizer
        self.training =  training

        # Preprocess, tokenize and store lengths
        processed_data = []
        for item in tqdm(data):
            formatted_instr, label_start_index = self.model.format_instruction(item, eval=not self.training)
            item['formatted_instruction'] = formatted_instr
            item['label_start_index'] = label_start_index
            tokenized_input = self.tokenizer(formatted_instr, truncation=True, return_tensors="pt")
            length = tokenized_input['input_ids'].size(1)  # Length of tokenized input
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

    def select(self, indices: list[int]):
        # Create a new dataset based on selected indices
        selected_data = [self.sorted_data[i] for i in indices]
        # Return a new instance of Tokenized_Sorted_Dataset with selected data
        selected_dataset = Tokenized_Sorted_Dataset([], self.model, self.training)
        selected_dataset.sorted_data = selected_data
        return selected_dataset
