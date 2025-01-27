'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from tqdm import tqdm
from vllm import SamplingParams
import torch
from models.evaluators.llm import BaseEval
import omegaconf
from models.evaluators.utils import process_llm_outputs_assess_scores, get_mean_without_unknown, unswitch_switched_scores, get_pairwise_scores_without_unknown, set_tq_description
import logging
logger = logging.getLogger(__name__)


class VLLMeval(BaseEval):
    """
    Evaluation class for vllm inference.
    """
    def __init__(self, model_config: dict, batch_size: int = None, config: str = "default_qa"):
        super().__init__(model_config, batch_size, config)

        eval_config = omegaconf.OmegaConf.load(f"config/evaluator/{config}.yaml")
        
        # VLLM-specific settings
        self.sampling_params = SamplingParams(
            best_of=1,
            temperature=0.0,
            top_p=1,
            top_k=-1,
            use_beam_search=False,
            max_tokens=eval_config['max_new_tokens'],
            presence_penalty=0,
            frequency_penalty=0,
            )
        
        self.llm.sampling_params.max_new_token = eval_config['max_new_tokens']
        self.batch_size = batch_size or self.llm.batch_size
        self.llm.max_new_tokens = eval_config['max_new_tokens']

    @torch.no_grad()
    def __call__(self, predictions, references, questions, opponent_predictions=None):
        assert len(predictions) == len(references) == len(questions)
        
        pairwise = (opponent_predictions is not None)
        options = self.options_pairwise if pairwise else self.options

        inputs = self.create_inputs(predictions=predictions, references=references, questions=questions, opponent_predictions=opponent_predictions)

        scores, weirds = [], []
        
        # Perform batch inference
        for i in (tq:=tqdm(range(0, len(inputs), self.batch_size), desc=f'LLM evaluation with {self.llm.model_name}...')):
            batch_examples = inputs[i:i+self.batch_size]

            instrs = [elt['instr'] for elt in batch_examples]
            
            decoded = self.llm.generate(instrs)
            
            batch_scores, batch_weird  = process_llm_outputs_assess_scores(decoded, options)

            if pairwise: # samples were randomly switched to avoid position bias: we unswitch!
                switches = [elt['switch'] for elt in batch_examples]
                batch_scores = unswitch_switched_scores(switched_scores=batch_scores, switches=switches)
                
            scores.extend(batch_scores)
            weirds.extend(batch_weird)
            
            set_tq_description(tq, scores, weirds, pairwise)
                
        logger.info(weirds)
        
        if pairwise:
            avg_scores = get_pairwise_scores_without_unknown(scores)
        else:
            avg_scores = get_mean_without_unknown(scores)
    
        return avg_scores, scores
