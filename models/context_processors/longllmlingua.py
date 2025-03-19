from models.context_processors.context_processor import ContextProcessor
from typing import List
from llmlingua import PromptCompressor 
from tqdm import tqdm

class LongLLMLingua(ContextProcessor):
    # https://llmlingua.com/llmlingua2.html
    def __init__(self, rate):
        # FIXME this next line should be commented when doing generation to avoid OOM
        self.model = PromptCompressor()
        self.name = f"longllmlingua_{rate}"
        self.model_name = self.name
        self.rate = rate
        self.predefined_context_processing_metrics = ["context_compression"]
        
    def _process(self, contexts: List[List[str]], queries: List[List[str]]):
        return [[self.model.compress_prompt(
                                            context,
                                            question=question,
                                            rate=self.rate,
                                            # Set the special parameter for LongLLMLingua
                                            condition_in_question="after_condition",
                                            use_sentence_level_filter=False,
                                            reorder_context="sort",
                                            dynamic_context_compression_ratio=0.3,
                                            keep_first_sentence=1,
                                            condition_compare=True,
                                            context_budget="+100",
                                            concate_question=False,
                                            token_budget_ratio=1.05,
                                            rank_method="longllmlingua",
                                        )["compressed_prompt"]]
                              for question, context in tqdm(zip(queries, contexts), 
                                                            desc='Compressing prompts with LongLLMLingua...', 
                                                            total=len(queries))], {}