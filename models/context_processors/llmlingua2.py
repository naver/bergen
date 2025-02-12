from models.context_processors.context_processor import ContextProcessor
from typing import List
from llmlingua import PromptCompressor
from tqdm import tqdm


class LLMLingua2(ContextProcessor):
    def __init__(self, model_name, rate, force_tokens=["\n", "?"]):
        self.model = PromptCompressor(model_name, use_llmlingua2=True)
        self.name = f"llmlingua2_rate{rate}"
        self.model_name = model_name
        self.rate = rate
        self.force_tokens = force_tokens

    def process(self, contexts: List[List[str]], queries: List[List[str]]):
        # llmlingua2 does not use queries
        return [
            [
                self.model.compress_prompt(
                    context_item, rate=self.rate, force_tokens=self.force_tokens
                )["compressed_prompt"]
                for context_item in context
            ]
            for context in tqdm(contexts, desc="Compressing prompts with LLMLingua2...")
        ]