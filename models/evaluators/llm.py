'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import GenerationConfig
import omegaconf
from tqdm import tqdm
import torch
from hydra.utils import instantiate
from models.evaluators.utils import process_llm_outputs_assess_scores, get_mean_without_unknown, unswitch_switched_scores, set_tq_description, get_pairwise_scores_without_unknown
import gc
import random


class BaseEval:
    """
    Base class for evaluation logic shared by LLMeval and VLLMeval.
    """
    def __init__(self, model_config: dict, batch_size: int = None, config: str = "default_qa"):
        """
        Base initializer for evaluation classes.
        """
        eval_config = omegaconf.OmegaConf.load(f"config/evaluator/{config}.yaml")
        model_config['init_args']['max_new_tokens'] = eval_config['max_new_tokens']

        self.llm = self.initialize_llm(model_config, eval_config)
        
        self.options = eval_config.output_options
        self.rubrik_section = ", ".join(self.options) 

        self.options_pairwise = eval_config.output_options_pairwise
        
        # Set up prompts
        self.prompt = eval_config['prompt']
        self.prompt_pairwise = eval_config['prompt_pairwise']
        self.system_prompt = self.prompt.system.format(rubrik_section=self.rubrik_section)
        self.system_prompt_pairwise =  self.prompt_pairwise.system.format(rubrik_section=self.rubrik_section)

        # Set up LLM parameters
        self.batch_size = batch_size or self.llm.batch_size
        self.llm.max_new_tokens = eval_config['max_new_tokens']
        
        # output_ids contains the token ids for the possible answers
        self.output_ids = [self.llm.tokenizer.encode(opt, add_special_tokens=False) for opt in sorted(self.options)]
        # output_values contain the associated 'score' for each option
        self.output_values = torch.tensor([self.options[opt] for opt in sorted(self.options)]).float()
        
        self.output_ids_pairwise = [self.llm.tokenizer.encode(opt, add_special_tokens=False) for opt in sorted(self.options_pairwise)]
        self.output_values_pairwise = torch.tensor([self.options_pairwise[opt] for opt in sorted(self.options_pairwise)]).float()

    def initialize_llm(self, model_config, eval_config):
        """
        Placeholder for LLM initialization, to be overridden by subclasses if needed.
        """
        return instantiate(model_config['init_args'], prompt=eval_config['prompt'])
    
    def __del__(self):
        torch.cuda.empty_cache()
        gc.collect()        

    def create_instruction(self, question, answer, prediction):
        prefix = []
        if getattr(self.llm.tokenizer, "chat_template") is not None and  'system' in self.llm.tokenizer.chat_template:
            prefix =  [{
                'role': 'system',
                'content': self.system_prompt
            }]
            prefix.extend([{
                'role': 'user',
                'content': self.prompt.user.format(rubrik_section=self.rubrik_section,
                                                    question=question,
                                                    answer=answer,
                                                    prediction=prediction)}]
            )
        
        else:
            prefix = ([{
                'role': 'user',
                'content': self.prompt.user_without_system.format(rubrik_section=self.rubrik_section,
                                                    question=question,
                                                    answer=answer,
                                                    prediction=prediction)
            }])
        if 'assistant' in self.prompt:
            prefix.extend([{'role': 'assistant',
                'content': self.prompt.assistant}]
            )
        return self.llm.tokenizer.apply_chat_template(prefix,  add_generation_prompt=True, tokenize=False) 
    
    def create_pairwise_instruction(self, question: str, answer: str, prediction_1: str, prediction_2: str) -> (str, bool):
        """
        To prevent positional bias, orders of answers is randomly switched
        We switch the scores appropriately later on in '__call__'
        so this method returns the prompt + the 'switch' boolean
        Unused arguments are used in the "eval"
        """
        switch = random.choice([True, False])
        if switch:
            prediction_1, prediction_2 = prediction_2, prediction_1
            
        assert hasattr(self.llm.tokenizer, 'chat_template'), 'Please use an LLM with a chat template'
        prefix =  [
                {'role': 'system', 'content': self.system_prompt_pairwise},
                {'role': 'user', 'content': self.prompt_pairwise.user.format(question=question, 
                                                                             answer=answer, 
                                                                             prediction_1=prediction_1, 
                                                                             prediction_2=prediction_2)}
            ]
        return self.llm.tokenizer.apply_chat_template(prefix,  add_generation_prompt=True, tokenize=False), switch
        
    def create_inputs(self, predictions, references, questions, opponent_predictions=None) -> dict:
        """
        Create all the prompts
        For pairwise case, it also creates the 'switches' which correspond to inversions in answer order to prevent bias.
        """
        assert len(predictions) == len(references) == len(questions)
        pairwise = (opponent_predictions is not None)
        if pairwise:
            assert len(opponent_predictions) == len(predictions)
            
        inputs = []
            
        for i in range(len(predictions)):
            if pairwise:
                sample_instr, sample_switch = self.create_pairwise_instruction(question=questions[i],
                                                                               answer=references[i],
                                                                               prediction_1=predictions[i],
                                                                               prediction_2=opponent_predictions[i])
                inputs.append({'instr': sample_instr, 'switch': sample_switch})
            else:
                sample_instr = self.create_instruction(question=questions[i], answer=references[i], prediction=predictions[i])
                inputs.append({'instr': sample_instr})
                
        return inputs

                
class LLMeval(BaseEval):
    """
    Evaluation class for HF inference.
    """
    def __init__(self, model_config: dict, batch_size: int = None, config: str = "default_qa"):
        super().__init__(model_config, batch_size, config)

        eval_config = omegaconf.OmegaConf.load(f"config/evaluator/{config}.yaml")
        self.use_logits = eval_config.use_logits

        # Set up generation config for HF
        self.generation_config = GenerationConfig.from_model_config(self.llm.model.config)
        self.generation_config.do_sample = False
        self.generation_config.max_new_tokens = self.llm.max_new_tokens   

    @torch.no_grad()
    def __call__(self, predictions, references, questions, opponent_predictions=None):
        """
        other_preditions: opponent model prediction in pairwise comparison
        """
        assert len(predictions) == len(references) == len(questions)
        
        pairwise = (opponent_predictions is not None)
        
        output_ids = self.output_ids_pairwise if pairwise else self.output_ids
        output_values = self.output_values_pairwise if pairwise else self.output_values
        options = self.options_pairwise if pairwise else self.options

        # list of dictionaries containing each sample formatted instruction, and switch (if pairwise)
        inputs = self.create_inputs(predictions=predictions, references=references, questions=questions, opponent_predictions=opponent_predictions)
            
        # The outputs are raw logits.
        scores, weirds = [], []
        # Perform batch inference
        for i in (tq:=tqdm(range(0, len(inputs), self.batch_size), desc=f'LLM evaluation with {self.llm.model_name}...')):
            # Extract batch
            batch_examples = inputs[i:i+self.batch_size]
            instrs = [elt['instr'] for elt in batch_examples]
            
            llm_inputs = self.llm.tokenizer(instrs, padding=True, truncation=True, return_tensors="pt")

            input_ids = llm_inputs['input_ids'].to(self.llm.model.device)
            attention_mask = llm_inputs['attention_mask'].to(self.llm.model.device)                        
                           
            if self.use_logits and not pairwise:
                self.generation_config.output_logits = True 
                self.generation_config.return_dict_in_generate=True                    
                model_outputs = self.llm.model.generate(input_ids, attention_mask=attention_mask, generation_config=self.generation_config)  
                
                #get processed logits from model outputs: expected shape (n_tokens, 1, vocab_size)
                model_scores = torch.stack(model_outputs.logits)
                #get scores corresponding to first token of predefined labels from the first generated tokens
                model_scores = model_scores[0, :, [tok[0] for tok in output_ids]].float()
                #normalizing scores - getting probablity of each of predefined labesl
                pos_prob = torch.softmax(model_scores, 1).detach().cpu()
                #final score is computed as interpolation between prob of label 
                # and its associated value (defined by options map in config): eg. p(x=yes)*1 + p(x=no)*0 

                for i, score in enumerate(pos_prob):
                    scores.append(torch.dot(score, output_values).item())
                                                
            else: # case: pairwise or pointwise, non-logits.
                output = self.llm.model.generate(
                        input_ids,                            
                        attention_mask=attention_mask,
                        generation_config=self.generation_config).detach().cpu().numpy()                
                decoded = self.llm.tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)
                
                batch_scores, batch_weirds  = process_llm_outputs_assess_scores(decoded, options)
                
                if pairwise:
                    # We post-process the scores to take into account the switches (which deter positional bias)
                    switches = [elt['switch'] for elt in batch_examples]
                    batch_scores = unswitch_switched_scores(switched_scores=batch_scores, switches=switches)
                    
                weirds.extend(batch_weirds)
                scores.extend(batch_scores)              
                
            set_tq_description(tq, scores, weirds, pairwise)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        if pairwise:
            avg_scores = get_pairwise_scores_without_unknown(scores)
        else:
            avg_scores = get_mean_without_unknown(scores)
            
        return avg_scores, scores
            