'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from tqdm import tqdm
import torch
from models.evaluators.llm import LLMeval
import omegaconf
from hydra.utils import instantiate
from models.evaluators.utils import process_llm_outputs_assess_scores, get_mean_without_unknown, unswitch_switched_scores
import logging
logger = logging.getLogger(__name__)


class VLLMeval(LLMeval):
    """
    - relies on vllm for inference, directly loads the model and runs inference (no need to initiate vllm server in advance) 
    - output score for each sample is 1 (when positive word is present in llm output) or 0  (otherwise) 
    """
    def __init__(self, model_config: dict, batch_size: int = None, config: str = "default_qa" ):
        """
            model_config: generator config specified as yaml file in cofig/generator directory
            batch_size: if none, it keeps default llm batch size from config 
            confg: name of evaluator config specified as yaml file at config/evaluators
        """
        eval_config = omegaconf.OmegaConf.load(f"config/evaluator/{config}.yaml")
        model_config['init_args']['max_new_tokens']= eval_config['max_new_tokens']
        self.llm = instantiate(model_config['init_args'], prompt=eval_config['prompt'])
        self.options = eval_config.output_options
        self.rubrik_section = ", ".join(["{"+opt+"}" for opt in self.options])
        self.prompt = eval_config['prompt']
        self.prompt_pairwise = eval_config['prompt_pairwise']

        self.llm.sampling_params.max_new_token = eval_config['max_new_tokens']
        self.llm.batch_size = batch_size or self.llm.batch_size
        self.llm.max_new_tokens = eval_config['max_new_tokens']
        self.system_prompt = eval(self.prompt.system).replace(':\ ', ': ')
        self.system_prompt_pairwise = eval(self.prompt_pairwise.system).replace(':\ ', ': ')
        self.output_ids = [self.llm.tokenizer.encode(opt, add_special_tokens=False)[-1] for opt in sorted(self.options)]
        self.output_values = torch.tensor([self.options[opt] for opt in sorted(self.options)]).float()

    @torch.no_grad()
    def __call__(self, predictions, references, questions, opponent_predictions=None):
        assert len(predictions) == len(references) == len(questions)
        
        pairwise = (opponent_predictions is not None)
        if not pairwise:
            assert len(opponent_predictions) == len(predictions)
            examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}
                        for i in range(len(predictions))]
        else:
            examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i], 'other_candidate': opponent_predictions[i]}
                        for i in range(len(predictions))]
        
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        if pairwise:
            instr, switches = []
            for sample in examples:
                sample_instr, sample_switch = self.create_pairwise_instruction(sample)
                instr.append(sample_instr)
                switches.append(sample_switch)
        else:
            instrs = [self.create_instruction(sample) for sample in examples]
            
        scores = list()
        weird = list() 
        # Perform batch inference
        for i in (tq:=tqdm(range(0, len(instrs), self.llm.batch_size), desc=f'LLM evaluation with {self.llm.model_name}...')):
            decoded = self.llm.generate(instrs[i:i+self.llm.batch_size])
            
            if pairwise:
                switched_scores, batch_weird  = process_llm_outputs_assess_scores(decoded, {'1': 1., '2': 0., '3': 0.5})
                batch_scores = unswitch_switched_scores(switched_scores=switched_scores, switches=switches[i:i+self.llm.batch_size])

            else:
                batch_scores, batch_weird  = process_llm_outputs_assess_scores(decoded, self.options)
                
            scores.extend(batch_scores)
            weird.extend(batch_weird)
                
            tq.set_description(f" score: {get_mean_without_unknown(scores)* 100:4.1f}%, weird :{float(len(weird))/len(scores)*100:4.1f}%")
        logger.info(weird)
        print("Weird", len(weird))
    
        return get_mean_without_unknown(scores), scores
