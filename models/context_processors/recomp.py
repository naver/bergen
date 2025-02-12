from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import T5ForConditionalGeneration
from models.context_processors.context_processor import ContextProcessor
import torch.nn.functional as F
import torch
from typing import List
from llmlingua import PromptCompressor
from tqdm import tqdm
import nltk
import string
import numpy as np

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class RecompExtractiveCompressor(ContextProcessor):
    def __init__(self, model_name="fangyuan/nq_extractive_compressor", batch_size=32, max_len=512, top_k=3, threshold=None, alway_select_title=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model_name = model_name
        if threshold is None: # default
            self.name = f"recomp_Ext_top{top_k}"
        else:
            self.name = f"recomp_Ext_t{threshold}"
        self.max_len = max_len
        self.top_k = top_k
        self.batch_size = batch_size
        self.alway_select_title = alway_select_title
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
                    context_sents_withtitle.append(sents_[0]+" "+s)
                    sents[(i, j)].append(s)
                    qis.append(i)
                    cis.append(j)
                    sis.append(k+1)
        toks = self.tokenizer(context_sents_withtitle, padding=True, truncation=True, return_tensors='pt', max_length=self.max_len)
        toks_q = self.tokenizer(queries, padding=True, truncation=True, return_tensors='pt', max_length=self.max_len)
        embs = []
        embs_q = []
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(context_sents_withtitle), self.batch_size), desc='Encoding contexts for pruning...'):
                input_ids = toks['input_ids'][batch_start: min(batch_start+self.batch_size, len(context_sents_withtitle))].to(self.device)
                attention_mask = toks['attention_mask'][batch_start: min(batch_start+self.batch_size, len(context_sents_withtitle))].to(self.device)
                token_type_ids = toks['token_type_ids'][batch_start: min(batch_start+self.batch_size, len(context_sents_withtitle))].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids) # what is token type ids???
                embeddings = mean_pooling(outputs[0], attention_mask).detach().cpu()
                embs.append(embeddings)
                
            for batch_start in tqdm(range(0, len(queries), self.batch_size), desc='Encoding queries for pruning...'):
                input_ids = toks_q['input_ids'][batch_start: min(batch_start+self.batch_size, len(queries))].to(self.device)
                attention_mask = toks_q['attention_mask'][batch_start: min(batch_start+self.batch_size, len(queries))].to(self.device)
                token_type_ids = toks_q['token_type_ids'][batch_start: min(batch_start+self.batch_size, len(queries))].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids) # what is token type ids???
                embeddings = mean_pooling(outputs[0], attention_mask).detach().cpu()
                embs_q.append(embeddings) 
                
        embs = torch.cat(embs)
        embs_q =  torch.cat(embs_q)

        scores = {qi:{ci:{} for ci in range(len(contexts[qi]))} for qi in range(len(queries))}
        for i, (qi, ci, si) in enumerate(zip(qis, cis, sis)):
            scores[qi][ci][si] = (embs_q[qi] @ embs[i]).item()
        selected_contexts = []
        for qi in range(len(queries)):
            selected_contexts_ = []
            for ci in range(len(contexts[qi])):
                scores_ = scores[qi][ci]
                if self.threshold is None: # default
                    selected_idxs = np.argsort(np.array([scores_[t+1] for t in range(len(scores_))]))[-self.top_k:]
                else:
                    selected_idxs = [t for t in range(len(scores_)) if scores_[t+1]>self.threshold]
                selected_cntx = " ".join([(titles[(qi, ci)]] if self.alway_select_title else [] \
                                          +[sents[(qi, ci)][idx] for idx in selected_idxs])
                selected_contexts_.append(selected_cntx)
            selected_contexts.append(selected_contexts_)
        return selected_contexts

class RecompAbstractiveCompressor(ContextProcessor):
    def __init__(self, model_name="fangyuan/nq_abstractive_compressor", batch_size=32, max_len=512, max_new_tokens=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model_name = model_name
        self.name = f"recomp_abskjek"
        self.max_len = max_len
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
    def process(self, contexts: List[List[str]], queries: List[List[str]]):
        inputs = []
        for q, cntxs in zip(queries, contexts):
            input_txt = "Question: {}\n Document: {}\n Summary: ".format(
                q,
                " ".join(cntxs),
            )  
            inputs.append(input_txt)
        # inputs = [prefix + inp for inp in inputs] # no prefix in recomp code?
        model_inputs = self.tokenizer(inputs, max_length=self.max_len, padding="max_length", truncation=True, return_tensors='pt',)
        selected_contexts = []
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(inputs), self.batch_size), desc='Pruning contexts...'):
                input_ids = model_inputs['input_ids'][batch_start: min(batch_start+self.batch_size, len(inputs))].to(self.device)
                attention_mask = model_inputs['attention_mask'][batch_start: min(batch_start+self.batch_size, len(inputs))].to(self.device)
                pred = self.model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           max_new_tokens=self.max_new_tokens)
                selected_contexts += [[self.tokenizer.decode(row, skip_special_tokens=True)] for row in pred]
        return selected_contexts