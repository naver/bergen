from models.context_processors.context_processor import ContextProcessor, get_compression
from transformers import AutoModel
from typing import List
import torch
from datasets.fingerprint import Hasher
import numpy as np

class ProvenceCompressor(ContextProcessor):
    # https://arxiv.org/abs/2501.16214
    def __init__(
        self,
        model_name,
        name = "provence",
        threshold=0.1,
        batch_size=32,
        always_select_title=True,
        enable_warnings=True,
        reorder=False,
        top_k=5,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.threshold = threshold
        self.batch_size = batch_size
        self.always_select_title = always_select_title
        self.enable_warnings = enable_warnings
        self.reorder = reorder
        self.top_k = top_k
        self.name = name+"_"+Hasher.hash(f"{threshold}{always_select_title}{reorder}")
        self.predefined_context_processing_metrics = [] 
    
    def _process(self, contexts: List[List[str]], queries: List[List[str]]):
        provence_out = self.model.process(queries,
                                  contexts,
                                  threshold=self.threshold,
                                  batch_size=self.batch_size,
                                  always_select_title=self.always_select_title,
                                  enable_warnings=self.enable_warnings,
                                  reorder=False
                                 )
        processed_contexts = provence_out["pruned_context"]
        if self.reorder:
            reranking_scores = provence_out["reranking_score"]
            # even though provence interface supports reordering,
            # we are re-implementing here
            # to be able to compute compression rate before reordering
            # (otherwise compression rate would include lower num of docs
            # and be not comparable to other methods)
        comps = []
        new_processed_contexts = []
        for i, (original, processed) in enumerate(zip(contexts, processed_contexts)):
            comp = get_compression(original, processed) 
            comps.append(comp)
            if self.reorder:
                idxs = np.argsort(reranking_scores[i])[::-1][:self.top_k]
                new_processed = [processed[j] for j in idxs]
                new_processed_contexts.append(new_processed)
            else:
                new_processed_contexts.append(processed)
        context_metrics = {"context_compression": np.mean(comps)}
        return new_processed_contexts, context_metrics