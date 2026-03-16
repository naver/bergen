'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import torch
import gc
from abc import ABC, abstractmethod
from modules.dataset import Tokenized_Sorted_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from jinja2.exceptions import TemplateError
from functools import partial
import random


class Generator(ABC):
    def __init__(self,
                 model_name: str = None,
                 batch_size: int = 1,
                 max_new_tokens: int = 1,
                 max_doc_len: int = 10**10,
                 max_length: int = None,
                 use_middle_truncation: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_doc_len = max_doc_len
        self.max_length = max_length
        self.use_middle_truncation = use_middle_truncation

    @abstractmethod
    def generate(self, inp):
        pass
    
    @abstractmethod
    def collate_fn(self, inp):
        pass

    def eval(self, dataset):
        with torch.no_grad():
            if self.tokenizer:
                tokenized_and_sorted_dataset = Tokenized_Sorted_Dataset(dataset, self, training=False)
                dataloader = DataLoader(tokenized_and_sorted_dataset, batch_size=self.batch_size, collate_fn=partial(self.collate_fn, eval=True), num_workers=4)
            else:
                dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=partial(self.collate_fn, eval=True), num_workers=4)
            
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
                
                torch.cuda.empty_cache()
                gc.collect()
                
            return query_ids, queries, instructions, responses, labels, ranking_labels

    def get_response(self):
        """
        This replaces the 'generation_prompt' in case the generator does not have a chat_template.
        It's used to prompt and also to identify the label positions to mask prompt in training.
        """
        return '\nResponse:\n'

    def get_response_template_ids(self):
        response_template =  self.get_response()
        return self.tokenizer.encode(response_template, add_special_tokens=False)
    
    def compile_prompt(self, system_prompt: str, user_prompt: str, question: str, docs: str = None, label: str = None):
            """
            Applying the chat template if it exists:
            NB: seemingly unused args are used in the 'eval' call.
            NB: if the label is not None, we assume training=True and then the answer of the model is added to the full prompt
            This method returns a tuple consisting of:
            - the final prompt
            - if a label is provided, the position of the first label index within the tokenized sequence (for masking in training)
            """
            # the prompt should finish with a generation prompt if we are in 'eval' mode i.e. when there is no label
            # NB: the generation prompt is empty (automatically included in the template rather) for llama/mistral/solar at least
            add_generation_prompt = (label is None) 
            
            label_start_index = None
            if self.tokenizer.chat_template is None:
                user_prompt_with_values = user_prompt.format(question=question, docs=docs)
                # we add the 'reponse incitation' to non chat-template
                prompt = f"{system_prompt}\n{user_prompt_with_values}" + self.get_response()
                if label is not None:
                    # Compute prompt size in tokens without labels.
                    label_start_index = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
                    prompt += label + self.tokenizer.eos_token

            else:        
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(question=question, docs=docs)}
                ]
                try:
                    # Handle the label
                    if label is not None:
                        # Compute the prompt without label, to know its length and hence where to mask in training
                        label_start_index = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, add_special_tokens=False))
                        messages.append({"role": "assistant", "content": label})
                    
                    prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)

                except TemplateError as e:
                    if "System role not supported" in str(e):
                        messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]

                        if label is not None:
                            # Compute the prompt without label, to know its length and hence where to mask in training  
                            label_start_index = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, add_special_tokens=False))
                            messages.append({"role": "assistant", "content": label})

                        prompt = self.tokenizer.apply_chat_template(messages,  add_generation_prompt=add_generation_prompt, tokenize=False)
                    else:
                        raise e
            
            
            if label is not None:
                assert label_start_index is not None # check we did find the prompt length
                if not prompt.endswith(self.tokenizer.eos_token):
                    prompt += self.tokenizer.eos_token # most models have this already, but not gemma-2b !
                
            return prompt, label_start_index

    def middle_truncation(self, docs):
        """
        Truncate documents by removing the middle section while preserving both the beginning and end.
        Args:
            docs (str): The document text to truncate
                
        Returns:
            str: The truncated document text
        """
        if docs is None or self.max_length is None or not hasattr(self, 'tokenizer'):
            return docs
        
        tokenized_docs = self.tokenizer(docs, truncation=False, return_tensors="pt")['input_ids'][0]
        docs_length = len(tokenized_docs)
        
        truncation_threshold = self.max_length - 128
        assert truncation_threshold >= 0, "Truncation threshold must be non-negative. Check max_length value."
        
        if docs_length > truncation_threshold:
            half = int(truncation_threshold / 2)
            
            first_half = tokenized_docs[:half]
            second_half = tokenized_docs[-half:]
            
            first_half_text = self.tokenizer.decode(first_half, skip_special_tokens=True)
            second_half_text = self.tokenizer.decode(second_half, skip_special_tokens=True)
            docs = first_half_text + second_half_text

        return docs


    def format_instruction(self, sample: dict, eval: bool = True) -> (str, int):
        """
        Makes the actual prompt from the prompt template and the model chat template
        Also return start index of the label in that prompt, if eval=True and a label is provided, None otherwise.
        If eval=True, then no label is added to the prompt.
        If eval=False, then the label is added to the prompt, for training (teacher forcing)
        """
        question = sample['query']
        label = None
        if not eval:
            label = (sample['label'] if isinstance(sample['label'], str) else random.choice(sample['label']))
            assert label is not None
        if 'doc' in sample:
            # We have retrieved documents:
            docs = ''
            input_docs = sample['doc']
            input_docs = [doc for doc in input_docs if len(doc.strip()) > 0]
            for i, doc in enumerate(input_docs):
                doc = ' '.join(doc.split()[:self.max_doc_len])
                docs += f"Document {i+1}: {doc}\n"
            if self.use_middle_truncation:
                docs = self.middle_truncation(docs)
            return self.compile_prompt(self.prompt.system, self.prompt.user, question, docs, label=label)
        else:
            # We have no retrieved documents: switch to no doc prompt
            return self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs, question, label=label)
