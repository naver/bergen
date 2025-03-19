from abc import ABC, abstractmethod
from typing import List, Dict, Union
import numpy as np
import warnings

class ContextProcessor(ABC):
    def __init__(self):
        self.predefined_context_processing_metrics = [] 
        # change in inherited classes if needed, e.g. "context_compression"
        # see options for metrics below in compute_predefined_context_processing_metrics
    
    @abstractmethod
    def _process(self, 
                 contexts: List[List[str]], 
                 queries: List[str]) -> List[List[str]]:
        pass
    
    def process(self, 
                contexts: List[List[str]], 
                queries: List[str]) -> Union[List[List[str]], Dict]:
        processed_contexts, context_metrics = self._process(contexts, queries)
        context_metrics.update(**self.compute_predefined_context_processing_metrics(queries, 
                                                                                     contexts,
                                                                                     processed_contexts)
                               )
        return processed_contexts, context_metrics

    def compute_predefined_context_processing_metrics(self, 
                                                      queries: List[str],
                                                      original_contexts: List[List[str]], 
                                                      processed_contexts: List[List[str]], 
                                                      ) -> Dict:
        computed_metrics = {}
        for metric_name in self.predefined_context_processing_metrics:
            if metric_name == "context_compression":
                computed_metrics["context_compression"] = np.mean([
                    get_compression(original, processed) 
                    for original, processed in 
                    zip(original_contexts, processed_contexts)
                ])
            else:
                warnings.warn(f"Undefined context processing metric: {metric_name}")
        return computed_metrics
                

def get_compression(original_contexts: List[str], compressed_contexts: List[str]) -> float:
    lo = len(original_contexts)
    lc = len(compressed_contexts)
    if lo != lc:
        warnings.warn("Context compression should be computed over lists of contexts of the same length! "+\
                      f"{lo} != {lc}")
    len_original = sum([len(cntx_original) for cntx_original in original_contexts])
    len_compressed = sum([len(cntx_compressed) for cntx_compressed in compressed_contexts])
    return (len_original-len_compressed)/len_original * 100