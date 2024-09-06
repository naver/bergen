'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import omegaconf
from tqdm import tqdm
import torch
import re
import numpy as np
from hydra.utils import instantiate
from models.evaluators.utils import *
import gc

class LLMeval():
    """
    - relies on default HF inference 
    - output score is a floating number corresponding to the logit score output by model.generate for pos_word
    """
    def __init__(self, model_config, batch_size=1, config="default_qa"):
        #model_config['init_args']['_target_'] = 'models.evaluators.llm.LLMeval'
        model_config = omegaconf.OmegaConf.load(f"config/generator/{model_config}.yaml")            
        eval_config = omegaconf.OmegaConf.load(f"config/evaluator/{config}.yaml")
        model_config['init_args']['max_new_tokens']= eval_config['max_new_tokens']

        self.use_logits = eval_config.use_logits
        self.llm = instantiate(model_config['init_args'], prompt=eval_config['prompt'])
        self.options = eval_config.output_options
        self.rubrik_section = ", ".join(["{"+opt+"}" for opt in self.options])
        self.prompt = eval_config['prompt']
        self.llm.max_new_tokens = eval_config['max_new_tokens']
        self.llm.batch_size = batch_size
        self.system_prompt = eval(self.prompt.system).replace(':\ ', ': ')
        #FIXME: what shall we do if label corrsponds to multiple tokens?
        self.output_ids = [self.llm.tokenizer.encode(opt, add_special_tokens=False) for opt in sorted(self.options)]
        self.output_values = torch.tensor([self.options[opt] for opt in sorted(self.options)]).float()
        
        self.generation_config = GenerationConfig.from_model_config(self.llm.model.config) 
        self.generation_config.do_sample=False,
        # according to documentation from https://huggingface.co/docs/transformers/v4.43.2/main_classes/text_generation this is supposed to force model to generate tokens from the list, but it doesn't seem to work in practice 
        # --> rollback to simple solution: just check first token logit of each predefined label
        self.generation_config.force_word_ids=self.output_ids, 
        self.generation_config.max_new_tokens=self.llm.max_new_tokens                    
                 
                                     
                
        
    def __del__(self):
    #    print(f"Delete evaluator {self.llm.model_name}")
        torch.cuda.empty_cache()
        gc.collect()        

    def create_instruction(self,sample):
        answer = sample['reference']
        question=sample['question']
        prediction=sample['candidate']
        if 'response' in sample:
            response = sample['response']
        else:
            response = None
        prefix = []
        if getattr(self.llm.tokenizer, "chat_template") is not None and  'system' in self.llm.tokenizer.chat_template:
            prefix =  [{'role': 'system',
                'content': self.system_prompt}]
            prefix.extend([{'role': 'user',
                'content': eval(self.prompt.user).replace(':\ ', ': ')}]
            )
        
        else:
            prefix = ([{'role': 'user',
                'content': eval(self.prompt.user_without_system).replace(':\ ', ': ')}]
            )
        if 'assistant' in self.prompt:
            prefix.extend([{'role': 'assistant',
                'content': eval(self.prompt.assistant).replace(':\ ', ': ')}]
                )
        if not response is None:
            prefix.extend([{'role': 'assistant',
                'content': response}]
            )
        return self.llm.tokenizer.apply_chat_template(prefix,  add_generation_prompt=True, tokenize=False) 

   
   
    def collate_fn(self, examples, max_length=512):
        instr = [self.create_instruction(sample) for sample in examples]  # Add prompt to each text
        instr_tokenized = self.llm.tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
        return instr_tokenized, instr

    @torch.no_grad()
    def __call__(self, predictions, references, questions):
        assert len(predictions) == len(references) == len(questions)
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        # The outputs are raw logits.
        scores = list()
        weird = list()
        # Perform batch inference
        full_inputs, full_instrs = self.collate_fn(examples)
        for i in (tq:=tqdm(range(0, len(examples), self.llm.batch_size), desc=f'LLM evaluation with {self.llm.model_name}...')):
            # Extract batch
            batch_examples = examples[i:i+self.llm.batch_size]
            inputs, instrs = self.collate_fn(batch_examples)
            input_ids = inputs['input_ids'].to(self.llm.model.device)
            attention_mask = inputs['attention_mask'].to(self.llm.model.device)                        
                           
            if self.use_logits:
                self.generation_config.output_logits=True 
                self.generation_config.return_dict_in_generate=True                    
                model_outputs = self.llm.model.generate(
                        input_ids,                            
                        attention_mask=attention_mask,
                        generation_config=self.generation_config
                )  
                #get processed logits from model outputs: expected shape (n_tokens, 1, vocab_size)
                model_scores = torch.stack(model_outputs.logits)
                #get scores corresponding to first token of predefined labels from the first generated tokens
                model_scores = model_scores[0, :, [tok[0] for tok in self.output_ids]].float()
                #normalizing scores - getting probablity of each of predefined labesl
                pos_prob = torch.softmax(model_scores, 1).detach().cpu()
                #final score is computed as interpolation between prob of label and it's associated value (defined by options map in config): eg. p(x=yes)*1 + p(x=no)*0 
                for i, score in enumerate(pos_prob):
                    scores.append(torch.dot(score,self.output_values))
            else:
                # discrete model output            
                # get real answer generation
                decoded = self.llm.generate(inputs)
                # #model_generations = self.llm.model.generate(input_ids,
                #                     attention_mask=attention_mask,
                #                     generation_config=self.generation_config 
                #                     )
                # decoded = self.llm.tokenizer.batch_decode(model_generations)
                # breakpoint()
                batch_scores, batch_weird  = process_llm_outputs_assess_scores(decoded, self.options)
                weird.extend(batch_weird)
                # if string value specified in options is present in the generated output: assign corresponding score,
                # if multiple values are present: take maximum value
                scores.extend(batch_scores)
                breakpoint()
            tq.set_description(f" score: {get_mean_without_unknown(scores)* 100:4.1f}%, weird :{float(len(weird))/len(scores)*100:4.1f}%")
        
        torch.cuda.empty_cache()
        gc.collect()
        return get_mean_without_unknown(scores), scores

