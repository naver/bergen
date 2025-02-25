from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import T5ForConditionalGeneration
from models.context_processors.context_processor import ContextProcessor
import torch.nn.functional as F
import torch
from typing import List
from tqdm import tqdm
import nltk
import string
import numpy as np


class DSLR_CE(ContextProcessor):
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", batch_size=32, max_len=512, threshold=None, always_select_title=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(self.device)
       
        
        self.model.eval()

        self.model_name = model_name
        if threshold is None: # default
            raise ValueError('DSLR threshold should not be none')
        else:
            self.name = f"dslrce_Ext_t{threshold}"
        self.max_len = max_len
    
        self.batch_size = batch_size
        self.always_select_title = always_select_title
        self.threshold = threshold
        
    def process(self, contexts: List[List[str]], queries: List[List[str]]):
        context_sents_withtitle = []
        sents = {}
        titles = {}
        qis = []
        cis = []
        sis = []
        for i, (q, docs) in enumerate(zip(queries, contexts)):
            for j, doc in enumerate(docs):
                sents_ = nltk.sent_tokenize(doc)
                titles[(i, j)] = sents_[0]
                sents[(i, j)] = []
                for k, s in enumerate(sents_[1:]):
                    context_sents_withtitle.append(q+" "+sents_[0]+" "+s)
                    sents[(i, j)].append(s)
                    qis.append(i)
                    cis.append(j)
                    sis.append(k+1)
        toks = self.tokenizer(context_sents_withtitle, padding=True, truncation=True, return_tensors='pt', max_length=self.max_len)
        probs = []
       
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(context_sents_withtitle), self.batch_size), desc='Encoding contexts for pruning...'):
                input_ids = toks['input_ids'][batch_start: min(batch_start+self.batch_size, len(context_sents_withtitle))].to(self.device)
                attention_mask = toks['attention_mask'][batch_start: min(batch_start+self.batch_size, len(context_sents_withtitle))].to(self.device)
                rank_score = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                ).logits
                # Use sigmoid since it's BCEWithLogitsLoss
                prob = torch.sigmoid(rank_score)
                probs+=(prob)

        scores = {qi:{ci:{} for ci in range(len(contexts[qi]))} for qi in range(len(queries))}
        for i, (qi, ci, si) in enumerate(zip(qis, cis, sis)):
            scores[qi][ci][si] = probs[i]
        selected_contexts = []
        for qi in range(len(queries)):
            selected_contexts_ = []
            for ci in range(len(contexts[qi])):
                scores_ = scores[qi][ci]
                
                selected_idxs = [t for t in range(len(scores_)) if scores_[t+1]>self.threshold]          
                selected_cntx = " ".join(( [titles[(qi, ci)]] if self.always_select_title else []) \
                                          +[sents[(qi, ci)][idx] for idx in selected_idxs])
                selected_contexts_.append(selected_cntx)
            selected_contexts.append(selected_contexts_)
        return selected_contexts
