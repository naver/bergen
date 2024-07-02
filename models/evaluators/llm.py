'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import omegaconf
from tqdm import tqdm
from models.generators.llm import LLM
import torch
import re
import numpy as np
class LLM:
    """
    - relies on default HF inference 
    - output score is a floating number corresponding to the logit score output by model.generate for pos_word
    """
    def __init__(self, model_name, batch_size=1, custom_format_instruction=None, pos_word="Yes", neg_word="No", prompt="default_prompt"):
        self.batch_size = batch_size
        quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='bfloat16',
        bnb_4bit_use_double_quant=False
        )
        self.pos_word = pos_word 
        self.neg_word = neg_word 
        
        self.custom_format_instruction = custom_format_instruction
        self.prompt = omegaconf.OmegaConf.load(f"config/evaluator/{prompt}.yaml")['prompt']
        self.system_prompt = eval(self.prompt.system)
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', quantization_config=quant_config, attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.model.config.use_cache = False
        self.model.eval()

        
        self.pos_tokenid, self.neg_tokenid = self.tokenizer.encode(f'\n{self.pos_word}', add_special_tokens=False)[-1], self.tokenizer.encode(f'\n{self.neg_word}', add_special_tokens=False)[-1]

   
    def create_instruction(self,sample):
        answer = sample['reference']
        question=sample['question']
        prediction=sample['candidate']
        pos_word = self.pos_word
        neg_word = self.neg_word
        prefix = []
        if 'system' in self.tokenizer.chat_template:
            prefix =  [{'role': 'system',
                'content': self.system_prompt}]
        prefix.extend([{'role': 'user',
            'content': eval(self.prompt.user)}]
            )
        prefix.extend([{'role': 'assistant',
            'content': eval(self.prompt.assistant)}]
            )
        return self.tokenizer.apply_chat_template(prefix,  add_generation_prompt=True, tokenize=False) 

   
   
    def collate_fn(self, examples, max_length=512):
        instr = [self.create_instruction(sample) if self.custom_format_instruction == None else self.custom_format_instruction(sample) for sample in examples]  # Add prompt to each text
        instr_tokenized = self.tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
        return instr_tokenized, instr

    @torch.no_grad()
    def __call__(self, predictions, references, questions):
        # Loading the TensorFlow Hub model
        assert len(predictions) == len(references) == len(questions)
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        
        inputs, instrs = self.collate_fn(examples)
        # The outputs are raw logits.
        scores = list()
        # Perform batch inference
        for i in tqdm(range(0, len(inputs['input_ids']), self.batch_size), desc=f'LLM evaluation with {self.model_name}...'):
            # Extract batch
            batch_input_ids = inputs['input_ids'][i:i+self.batch_size].to('cuda')
            batch_attention_masks = inputs['attention_mask'][i:i+self.batch_size].to('cuda')
            instr_batch = instrs[i:i+self.batch_size]
            # discrete model output 
            # generated_tokens = self.model.generate(input_ids=batch_input_ids, attention_mask=batch_attention_masks, max_new_tokens=3) 
            # generated_answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # for generated_answer in generated_answers:
            #     answer = re.findall(r'\{([^}]+)\}', generated_answer)[-1]
            #     scores.append(1 if answer == 'equivalent' else 0)
            # continuous model output:
            model_scores = self.model.generate(input_ids=batch_input_ids, attention_mask=batch_attention_masks, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True).scores
            model_scores = torch.stack(model_scores)
            model_scores = model_scores[0, :, [self.neg_tokenid, self.pos_tokenid]].float()
            pos_prob = torch.softmax(model_scores, 1)[:, 1].detach().cpu()
            for i, score in enumerate(pos_prob):
                scores.append(score.float())

        torch.cuda.empty_cache()
        return np.mean(scores), scores

